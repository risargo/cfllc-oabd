import sys
import os
import numpy as np
import cplex
from cplex import SparsePair
import json
import time
import traceback
import datetime


def main(filename):
    print(__file__)
#    assert len(sys.argv) > 1,'please, provide a dat file'
#    filename = sys.argv[1]
    filenm = filename   
    dt = Data(filenm)
    create_model(dt)
    name,objective,gap,nodes,time = solve_callback_model(dt)
    return name,objective,gap,nodes,time

class CutsCallback():    
    def __init__(self,data):
      self.data = data
      self.cutcoefs = np.zeros(self.data.nj+1)
      self.cutinds = np.zeros(self.data.nj+1,dtype='int')
      self.nogoodcoefs = np.zeros(self.data.ni + self.data.nj,dtype = 'float')      
      self.nogoodinds = np.zeros(self.data.ni + self.data.nj,dtype='int')
      self.nogoodinds[self.data.z] = np.array(self.data.z)
      self.nogoodinds[self.data.x] = self.data.x
      self.cutinds[1:] = self.data.x
      self.nogoodcoefs[self.data.z] = 1     
      
    def separate_lazy_cut(self,context):
        I = range(self.data.ni)
        J = range(self.data.nj)
        data = self.data
        cpx = self.data.cpx

        _x = np.array(context.get_candidate_point(self.data.x.tolist()))
        _z = context.get_candidate_point(self.data.z)
        cuts = []
        rhss = []
        
        is_feasible = True
        total_phi = 0.0
        for i in I:
            gamma = data.gamma[i]
            xpi = np.take(_x,data.sorted_idpi[i])

            Jtilde = data.sorted_idpi[i][xpi > 1e-6] 
         
            _g = 0 if Jtilde.size == 0 else data.pi[i][Jtilde[:gamma]].sum() 
            _phi = data.b[i]/(_g + 1)         
            
            total_phi += _phi
                 
            if  _phi - _z[i] > 1e-4:
            
               is_feasible = False
               if Jtilde.size < gamma: 
                  _ll = [(data.b[i]*data.pi[i][j])/(_g + 1)**2 for j in J] 
               else: 
                  J0 = Jtilde[gamma:] 
                  pihat = 0 if J0.size == 0 else data.pi[i][J0[0]] 
                  _ll = [0.0 if j in J0 else (data.b[i] * max(data.pi[i][j] - pihat,0.0))/(_g + 1)**2 for j in J] 
                  
               rhs = _phi + np.dot(_ll,_x)
               
               self.cutcoefs[0] = 1
               self.cutinds[0] = data.z[i]
               self.cutcoefs[1:] = _ll
               
               cuts.append(SparsePair(ind=self.cutinds.tolist(),val=self.cutcoefs.tolist()))
               rhss.append(rhs)
                           
        if len(cuts) > 0:     
           context.reject_candidate(constraints = cuts,senses=["G"]*len(cuts),rhs = rhss)
           
        if is_feasible == False:
           # add no good cut
           posx = np.where(_x >= 0.5)
           negx = np.where(_x < 0.5)

           self.nogoodcoefs[negx[0]] = total_phi
           self.nogoodcoefs[posx[0]] = 0
           rhs = total_phi            
           
           context.reject_candidate(constraints = [SparsePair(ind=self.nogoodinds.tolist(),val=self.nogoodcoefs.tolist())], senses = ["G"],rhs = [rhs])
           #context.reject_candidate(constraints = None, senses = None,rhs = None)

           
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

def solve_callback_model(data):
    I = range(data.ni)
    J = range(data.nj)
    
    cpx = data.cpx
    cb = CutsCallback(data)

    contextmask = cplex.callbacks.Context.id.candidate
    #contextmask |= cplex.callbacks.Context.id.relaxation
    #contextmask = cplex.callbacks.Context.id.thread_up
    #contextmask |= cplex.callbacks.Context.id.thread_down
    cpx.set_callback(cb, contextmask)

    cpx = data.cpx
    
    startt = time.time()
    cpx.solve()
    rtime = time.time() - startt 
    print("datafile   :{:18s}".format(data.filename))
    print("Objective  : {:18.2f}".format(cpx.solution.get_objective_value()))
    print("Status     : {:>18s} ".format(cpx.solution.status[cpx.solution.get_status()]))
    print("RMIPgap (%): {:18.8f}".format(100*cpx.solution.MIP.get_mip_relative_gap()))
    print("# bb nodes : {:18d}".format(cpx.solution.progress.get_num_nodes_processed()))
    print("Time (s)   : {:18.2f}".format(rtime))
    print()
   
    return data.filename,cpx.solution.get_objective_value(),100*cpx.solution.MIP.get_mip_relative_gap(),cpx.solution.progress.get_num_nodes_processed(),rtime
              
def create_model(data):

    ni,nj = data.ni,data.nj
    I,J = range(ni),range(nj)
    
    cpx = cplex.Cplex()

    cpx.parameters.threads.set(1)    
    cpx.parameters.timelimit.set(7200.0)
    cpx.parameters.mip.tolerances.mipgap.set(1e-3)
    cpx.parameters.mip.tolerances.integrality.set(1e-9)
    #cpx.set_results_stream(None)
    #cpx.set_log_stream(None)
    #cpx.set_warning_stream(None)
    #num_threads = 1;
    #cpx.parameters.threads.set(num_threads)
    #cpx.parameters.timelimit.set(600)
    #cpx.parameters.mip.tolerances.mipgap.set(1e-8)
   
    x = np.array(cpx.variables.add(obj=data.f.tolist(),lb=[0.0] * nj,ub=[1.0] * nj,types=['B'] * nj,names=['x('+str(j)+')' for j in J]),dtype='int')
    z = list(cpx.variables.add(obj=[1.0]*ni,lb=[0.0] * ni,ub=[cplex.infinity] * ni,types=['C'] * ni,names=['z('+str(i)+')' for i in I]))   
    
    cpx.objective.set_sense(cpx.objective.sense.minimize)

    data.cpx = cpx
    data.x = x
    data.z = z

class Data():
   def __init__(self,filename):
       self.filename = filename
       self.read_data_file()
           
   def read_data_file(self):
       assert os.path.isfile(self.filename) == True, 'please, provide a valid data file'
       with open(self.filename) as f:
            dt = json.load(f)
            self.ni = dt['ni']
            self.nj = dt['nj']
            self.gamma = dt['gamma']
            self.cxi = np.array(dt['cxi'])
            self.cyi = np.array(dt['cyi'])
            self.cxj = np.array(dt['cxj'])
            self.cyj = np.array(dt['cyj'])
            self.b = np.array(dt['b'])
            self.f = np.array(dt['f'])
            self.pi = np.array(dt['pi'])
            self.sorted_idpi = np.fliplr(np.argsort(self.pi,axis=1))
                
if __name__ == "__main__":
    st = datetime.datetime.utcnow()-datetime.timedelta(hours=3)
    print("starting time: ", st)
    assert len(sys.argv) > 1,'please, provide a dat file'
    filename = sys.argv[1]    
    main(filename)
    et = datetime.datetime.utcnow()-datetime.timedelta(hours=3)
    print("Ending time: ", et)
    print("Time difference: ",et-st)    