import sys
import os
import traceback
import numpy as np
import cplex
from cplex import SparsePair

import json
from time import time
import datetime
from dotmap import DotMap


#def main():
    #assert len(sys.argv) > 1,'please, provide a dat file'
    #filename = sys.argv[1]
    #solve_model(dt)

def main(filename):    
    dt = read_data_file(filename) #função para leitura de instâncias.
    create_model(dt)       
    ub,lb,gap,timee,nodes,strstatus = solve_model(dt)
    
    print("LB         : {:18.2f}".format(lb))
    print("UP         : {:18.2f} ".format(ub))
    print("RMIPgap (%): {:18.8f}".format(gap))
    print("Nodes      : {:18d} ".format(nodes))
    print("Time (s)   : {:18.2f}".format(timee))
    print("Status     : {:s}".format(strstatus))
    print()
    
    return lb,ub,gap,timee,nodes
    
class CutsCallback():    
    def __init__(self,data):
      self.data = data

    def invoke(self, context):
        try:
           if context.in_relaxation():
              self.separate_user_cut(context)
           elif context.in_candidate():
              self.separate_lazy_cut(context)
        except:
           info = sys.exc_info()
           print('#### Exception in callback: ', info[0])
           print('####                        ', info[1])
           print('####                        ', info[2])
           traceback.print_tb(info[2], file=sys.stdout)
           raise      

    def separate_user_cut(self,context):
        pass

    def separate_lazy_cut(self,context):
        I = range(self.data.ni)
        J = range(self.data.nj)
        
        _x = np.array(context.get_candidate_point(self.data.x))
        _z = context.get_candidate_point(self.data.z)
        _y = [context.get_candidate_point(self.data.y[i]) for i in I]
       
        is_cut, cuts, rhss = self.separate_oa(_x,_y,_z)
        if is_cut == True:
           context.reject_candidate(constraints = cuts,senses=["G"]*len(cuts),rhs = rhss) 
           
    def separate_oa(self,_x,_y,_z):
        I = range(self.data.ni)
        J = range(self.data.nj)
        is_cut = False
        
        cuts = []
        rhss = []
        
        llambda = [1.0] * (self.data.nj + 1) 
        cutinds = [-1] * (self.data.nj + 1)
        
        b,pi = self.data.b,self.data.pi
        z,y = self.data.z,self.data.y

        for i in I:
        
           denominador = 1 + sum(pi[i][j] * _y[i][j] for j in J) 
           _fi = b[i]/denominador
           if _fi - _z[i] > 1e-4:
              is_cut = True
              cutinds[self.data.nj] = self.data.z[i]
              rhsi = _fi
              for j in J:
                 cutinds[j] = y[i][j]    
                 grad = -(b[i] * pi[i][j]) / (denominador*denominador) 
                 llambda[j] = -grad
                 rhsi -= grad * _y[i][j]
             
              cuts.append(SparsePair(ind=cutinds.copy(),val=llambda.copy()))
              rhss.append(rhsi)
       
        return is_cut,cuts,rhss
            
    
def solve_model(data):
    I = range(data.ni)
    cpx = data.cpx
    nodes = 0
    gap = 100.0
    rtime = 0
    lb = 0
    ub = float("inf")
    startt = time()
    
    cb = CutsCallback(data)
    contextmask = cplex.callbacks.Context.id.candidate
    cpx.set_callback(cb, contextmask)

    cpx.solve()

    status = cpx.solution.get_status()    
    strstatus = cpx.solution.get_status_string()
    if status == cpx.solution.status.optimal\
    or status == cpx.solution.status.feasible\
    or status == cpx.solution.status.optimal_tolerance\
    or status == cpx.solution.status.MIP_optimal\
    or status == cpx.solution.status.MIP_time_limit_feasible\
    or status == cpx.solution.status.MIP_dettime_limit_feasible\
    or status == cpx.solution.status.MIP_abort_feasible\
    or status == cpx.solution.status.MIP_optimal_infeasible\
    or status == cpx.solution.status.MIP_feasible:
    
       nodes = cpx.solution.progress.get_num_nodes_processed()
       ub = cpx.solution.get_objective_value()
       lb = cpx.solution.MIP.get_best_objective()
       gap = 100.0 *cpx.solution.MIP.get_mip_relative_gap()
       
    elif status == cpx.solution.status.MIP_infeasible\
    or status == cpx.solution.status.infeasible\
    or status == cpx.solution.status.unbounded\
    or status == cpx.solution.status.MIP_time_limit_infeasible\
    or status == cpx.solution.status.MIP_dettime_limit_infeasible\
    or status == cpx.solution.status.MIP_abort_infeasible\
    or status == cpx.solution.status.MIP_unbounded:
       print("otimizacao falhou")
       
       
    rtime = round(time() - startt,2) 
        
    
             

    return ub,lb,gap,rtime,nodes,strstatus             
                       
def create_model(data):

    ni,nj = data["ni"],data["nj"]
    I,J = range(ni),range(nj)
    
    cpx = cplex.Cplex()
    
    cpx.parameters.threads.set(1)
    cpx.parameters.timelimit.set(7200.0)
    cpx.parameters.mip.tolerances.mipgap.set(1e-3)
    cpx.parameters.mip.tolerances.integrality.set(1e-9)
    #cpx.set_results_stream(None)
    #cpx.set_log_stream(None)
    #cpx.set_warning_stream(None)
    
    x = list(cpx.variables.add(obj=data.f,lb=[0.0] * nj,ub=[1.0] * nj,types=['B'] * nj,names=['x('+str(j)+')' for j in J]))
    z = list(cpx.variables.add(obj=[1.0]*ni,lb=[0.0] * ni,ub=[cplex.infinity]*ni,types=['C'] * ni,names=['z('+str(i)+')' for i in I]))
    y = [ list(cpx.variables.add(obj=None,lb=[0.0] * nj,ub=[1.0] * nj, types=['C'] * nj,names=['y('+str(i)+','+str(j)+')' for j in J])) for i in I ]
    
    cpx.objective.set_sense(cpx.objective.sense.minimize)
                    
    for i in I:
       for j in J:
           cpx.linear_constraints.add(
               lin_expr=[cplex.SparsePair(
                   ind=[y[i][j],x[j]], val=[1.0,-1.0])],
               senses=["L"],
               rhs=[0.0])
            
    for i in I:
       cpx.linear_constraints.add(
           lin_expr=[cplex.SparsePair(
               ind=y[i], val=[1.0] * nj)],
           senses=["L"],
           rhs=[data.gamma[i]])
    
    #cpx.write("oacb.lp")            
    data.cpx = cpx
    data.x = x
    data.y = y
    data.z = z
                          
def read_data_file(filename):
    f = open(filename)
    dt = json.load(f)
    data = DotMap(dt)
    f.close()
    return data
    
if __name__ == "__main__":
   #st = datetime.datetime.utcnow()-datetime.timedelta(hours=3)
   #print("starting time: ", st)
   assert len(sys.argv) > 1,'please, provide a dat file'
   filename = sys.argv[1]     
   main(filename)
   #main()
   #et = datetime.datetime.utcnow()-datetime.timedelta(hours=3)    
   #print("Ending time: ", et)
   #print("Time difference: ",et-st)