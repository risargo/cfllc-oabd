import sys
import os
import traceback
import numpy as np
from numpy import newaxis
import cplex
from cplex import SparsePair
import json
from time import time
import datetime
from itertools import  product
from scipy.spatial.distance import cdist
from math import ceil, floor
maximumrunningtime = 7200.0
heurmaximumrunningtime = 60

class CRoundingHeur:
    def __init__(self,dt,_x):
        self.data = dt
        ix = np.argsort(_x)[::-1]
        x = np.zeros(dt.nj).astype(int)
        self.z = np.zeros(dt.ni).astype(float)
        ub = float('inf')
        for j in ix:
            x[j] = 1
            inf = self.get_obj(x) 
            if ub > inf:
               ub = inf
            else:
               x[j] = 0
               break
        self.ub = ub
        self.x = x
        self.ub = self.vnd(ub)

    def n0_close_facilities(self,obj):
        J1 = np.argwhere(self.x>0.9)[:,0]
        for j in J1:
            self.x[j] = 0
            newobj = self.get_obj(self.x)
            if newobj >= obj:
                self.x[j] = 1
            else:
                obj = newobj
        return obj

    def n1_open_facilities(self,obj):
        J0 = np.argwhere(self.x<0.1)[:,0]

        for j in J0:
            self.x[j] = 1
            newobj = self.get_obj(self.x)
            if newobj >= obj:
                self.x[j] = 0
            else:
                obj = newobj

        return obj

    def n2_swap(self,obj):
        dt = self.data
        nhalf = ceil(dt.nj/4)
        neighbor = dt.neighbor

        J0 = np.argwhere(self.x<0.1)[:,0]
        
        for j0 in J0:
            #J1 = np.argwhere(self.x>0.9)[:,0]
            #for j1 in J1:
            for j1 in neighbor[j0][:nhalf]: 
                if self.x[j1] > 0.9:
                   self.x[j0],self.x[j1] = 1,0
                   newobj = self.get_obj(self.x)
                   if newobj >= obj:
                      self.x[j0],self.x[j1] = 0,1
                   else:
                       obj = newobj
                       break
        return obj

    def vnd(self,obj):
        bestx = np.copy(self.x)
        bestobj = obj
        h = 0
        startingtime = time()
        while (h < 3):
            if h == 0:
               obj = self.n0_close_facilities(obj)
            elif h == 1:
               obj = self.n1_open_facilities(obj)
            elif h == 2:
               obj = self.n2_swap(obj)

            if obj < bestobj:
               np.copyto(bestx,self.x)
               bestobj = obj
               if h < 3:
                  h = 0
            else:
               h += 1

    def get_obj(self,x):
        dt = self.data
        I,J = range(dt.ni),range(dt.nj)
        totalcost = (x*dt.f).sum()
        for i in I:
            gamma = dt.gamma[i]
            den = 1.0
            for j in dt.sorted_idpi[i]:
                if x[j] == 1:
                   gamma -= 1
                   den += dt.pi[i][j]
                   if gamma == 0:
                      break
            totalcost += dt.b[i] / den
            self.z[i] = den - 1.0
        return totalcost
        
def oabdhotheur2(data):
    global time
    #assert len(sys.argv) > 1,'please, provide a dat file'
    #filename = sys.argv[1]    
    #dt = Data(filename)
    dt = data#CData(number_of_customers=300,number_of_facilities=100,u_scale_factor=1.e6,gamma=2,is_uniform_gamma=True)    
    #dt.nprecuts = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    #dt.nprecuts = 25 #int(dt.nj/10)
    dt.nprecuts = int(dt.nj/10)
    #dt.nhotstartloops = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    #dt.nhotstartloops = 20 #int(dt.ni/10) 
    dt.nhotstartloops = int(dt.ni/10) 
    create_model(dt)
    
    sub = Subproblem(dt)
    
    st = time()

    ttime = 0.0
    
    if dt.nprecuts > 0:
       print('separating %d rounds of oa cuts' % dt.nprecuts)
       add_oa_initial_cuts(dt,sub)
    
    if dt.nhotstartloops > 0:
       ttime,_x = hot_start(dt,sub,dt.nhotstartloops,st)
       # call heuristic VND
       if 1: 
          stheur = time()

          #rheur = CRoundingHeur(dt,_x)

          #heur = CHeur(dt,rheur.x)
          heur = CHeur(dt,_x)
          heur.run()
          etheur = time()
          print(f'heur time: {etheur-stheur:.2f} s')
          # set integer feasible solution 
          # set_master_problem_initial_integer_feasible_solution(dt,heursol)
          set_heur_oa_and_bd_cuts(dt,heur.x,heur.z,sub)

    name,objective,gap,nodes,timee = solve_callback(dt,sub,maximumrunningtime - ttime)        
    
    return name,objective,gap,nodes,timee

def set_master_problem_initial_integer_feasible_solution(data,sol):
    cpx = data.cpx
    #indices = np.argwhere(_x > 1e-6)[:,0].tolist()
    #nindices = len(indices)
    newsol = cpx.MIP_starts.add(
            SparsePair(ind= data.x + data.g, val= sol['x'].tolist() + sol['g'].tolist()),
            cpx.MIP_starts.effort_level.auto)
       
def set_heur_oa_and_bd_cuts(data,_x,_z,sub):
      cpx = data.cpx
      is_cut1, cuts, rhss,total_phi = sub.separate_oa_cuts(_z,None)
      if is_cut1 == True:
           cpx.linear_constraints.add(lin_expr=cuts,senses = ["G"]*len(cuts),rhs = rhss)        

      is_cut, cuts, rhss = sub.separate_bd_cuts_analytically(_x,_z)
      if is_cut == True:
           cpx.linear_constraints.add(lin_expr=cuts,senses = ["G"]*len(cuts),rhs = rhss)        

class CHeur():
    def __init__(self,data,_x):
        self.data = data
        self._x = _x
        self.x = np.zeros(_x.shape).astype(int)
        self.z = np.zeros(data.ni).astype(float)

    def run(self):
        dt = self.data
        st = time()
        obj = self.create_initial_solution()
        print(f'heur 0: {obj:18.2f}')
        et = time()
        print(f'rounding time: {et-st:18.2f}')
        return self.vnd(obj) 

    def check_time(self,startingtime,h):
        if time() - startingtime > heurmaximumrunningtime:
           return 4
        else:
           return h

    def vnd(self,obj):
        bestx = np.copy(self.x)
        bestobj = obj
        h = 0
        startingtime = time()
        times = [0.0] * 3
        while (h < 3):
            st = time()
            if h == 0:
               obj = self.n0_close_facilities(obj)
               h = self.check_time(startingtime,h)
            elif h == 1:
               obj = self.n1_open_facilities(obj)
               h = self.check_time(startingtime,h)
            elif h == 2:
               obj = self.n2_swap(obj)
               h = self.check_time(startingtime,h)
            et = time()
            times[h] += (et - st)
            if obj < bestobj:
               np.copyto(bestx,self.x)
               bestobj = obj
               if h < 3:
                  h = 0
            else:
               h += 1
        print(f'heurf {bestobj:12.2f}')
        print(times)
        return bestobj

    def n0_close_facilities(self,obj):
        J1 = np.argwhere(self.x>0.9)[:,0]
        for j in J1:
            self.x[j] = 0
            newobj = self.eval_objective_function(self.x)
            if newobj >= obj:
                self.x[j] = 1
            else:
                obj = newobj
        return obj

    def n1_open_facilities(self,obj):
        J0 = np.argwhere(self.x<0.1)[:,0]
        for j in J0:
            self.x[j] = 1
            newobj = self.eval_objective_function(self.x)
            if newobj >= obj:
                self.x[j] = 0
            else:
                obj = newobj
        return obj

    def n2_swap(self,obj):
        dt = self.data
        nhalf = ceil(dt.nj/4)
        neighbor = dt.neighbor

        J0 = np.argwhere(self.x<0.1)[:,0]
        
        for j0 in J0:
            #J1 = np.argwhere(self.x>0.9)[:,0]
            #for j1 in J1:
            for j1 in neighbor[j0][:nhalf]: 
                if self.x[j1] > 0.9:
                   self.x[j0],self.x[j1] = 1,0
                   newobj = self.eval_objective_function(self.x)
                   if newobj >= obj:
                      self.x[j0],self.x[j1] = 0,1
                   else:
                       obj = newobj
                       break
        return obj

    def create_initial_solution(self):
        _x = self._x
        self.x[:] = np.rint(_x)

        self.J = np.argwhere(_x > 1e-6)[:,0]
        obj = self.eval_objective_function(self.x)

        return obj
    
    def eval_objective_function(self,x):
        dt = self.data
        I,J = range(dt.ni),range(dt.nj)
        totalcost = (x*dt.f).sum()
        for i in I:
            gamma = dt.gamma[i]
            den = 1.0
            for j in dt.sorted_idpi[i]:
                if x[j] == 1:
                   gamma -= 1
                   den += dt.pi[i][j]
                   if gamma == 0:
                      break
            totalcost += dt.b[i] / den
            self.z[i] = den - 1
        return totalcost

class BDCutsCallback():    
    def __init__(self,data,sub):
      self.data = data
      self.sub = sub
      
      self.nogoodcoefs = np.zeros(data.ni + data.nj,dtype = 'float')      
      self.nogoodinds = np.zeros(data.ni + data.nj,dtype='int')
      self.nogoodinds[data.z] = np.array(data.z)
      self.nogoodinds[data.x] = data.x
      self.nogoodcoefs[data.z] = 1     

    def invoke(self, context):
        try:
           if context.in_relaxation():
              self.separate_user_cut(context)
           elif context.in_candidate():
              self.separate_bd_cuts(context)
        except:
           info = sys.exc_info()
           print('#### Exception in callback: ', info[0])
           print('####                        ', info[1])
           print('####                        ', info[2])
           traceback.print_tb(info[2], file=sys.stdout)
           raise 
                 
    def separate_bd_cuts(self,context):
      dt = self.data
      sub = self.sub
      
      _x = np.array(context.get_candidate_point(dt.x))
      _g = context.get_candidate_point(dt.g)
      _z = context.get_candidate_point(dt.z)      
      
      is_cut1, cuts, rhss,total_phi = sub.separate_oa_cuts(_g,_z)
      if is_cut1 == True:
         context.reject_candidate(constraints = cuts,senses=["G"]*len(cuts),rhs = rhss)  
         
      is_cut, cuts, rhss = sub.separate_bd_cuts_analytically(_x,_g)
      if is_cut == True:
         context.reject_candidate(constraints = cuts,senses=["G"]*len(cuts),rhs = rhss)  
      
      #is_cut, cuts, rhss = sub.separate_bd_cuts_via_lp(_x,_g)
      #if is_cut == True:
      #   context.reject_candidate(constraints = cuts,senses=["G"]*len(cuts),rhs = rhss)  
      # no good cut   
      #if is_cut == False:
      #   posx = np.where(_x >= 0.5)
      #   negx = np.where(_x < 0.5)

      #   self.nogoodcoefs[negx[0]] = total_phi
      #   self.nogoodcoefs[posx[0]] = 0
      #   rhs = total_phi            
      #   context.reject_candidate(constraints = [SparsePair(ind=self.nogoodinds.tolist(),val=self.nogoodcoefs.tolist())], senses = ["G"],rhs = [rhs])

             
class Subproblem():
    def __init__(self,data):
        self.data = data

        ni,nj = self.data.ni,self.data.nj

        self.cutcoefs = np.zeros(nj+1)
        self.cutinds = np.zeros(nj+1,dtype='int')
        
        self.cutinds[1:] = self.data.x
        
        self.create_model()

    def create_model(self):
        ni,nj = self.data.ni,self.data.nj
        I,J = range(ni),range(nj)

        self.cpx = cplex.Cplex()

        self.cpx.parameters.timelimit.set(7200.0)
        self.cpx.parameters.threads.set(1)
        #self.cpx.parameters.mip.tolerances.mipgap.set(1e-3)
        #self.cpx.parameters.mip.tolerances.integrality.set(1e-9)
        #self.cpx.set_results_stream(None)
        #self.cpx.set_log_stream(None)
        #self.cpx.set_warning_stream(None)

        self.cpx.objective.set_sense(self.cpx.objective.sense.maximize)
        self.cpx.set_problem_type(self.cpx.problem_type.LP)
        
        self.s = list(self.cpx.variables.add(obj = None, lb = [0.0], ub=[1.0], types = ['C'], names=['s']))[0]
        self.u = list(self.cpx.variables.add(obj = None, lb = [0.0], ub=[cplex.infinity], types = ['C'], names=['u']))[0]
        self.v = list(self.cpx.variables.add(obj = None, lb = [0.0] * nj, ub=[cplex.infinity] * nj, types = ['C'] * nj, names=["v(%d)" % j for j in J])) 
        #self.r = list(self.cpx.variables.add(obj = [-1.0] * nj, lb = [0.0] * nj, ub=[cplex.infinity] * nj, types = ['C'] * nj, names=["r(%d)" % j for j in J]))         

        #constrs = [SparsePair(ind=[self.u,self.r[j]] + self.v,val=[-1.0] * (nj + 2))  for j in J]
        constrs = [SparsePair(ind=[self.s,self.u,self.v[j]],val=[1.0,-1.0,-1.0])  for j in J]
        self.cnstr = list(self.cpx.linear_constraints.add(lin_expr=constrs, senses = ["L"]*len(constrs), rhs = [0.0]*len(constrs)))

    def separate_bd_cuts_via_lp(self,_x,_g):
        ni,nj = self.data.ni, self.data.nj
        I, J = range(ni), range(nj)
    
        is_cut = False
        cuts = []
        rhss = []
        for i in I:
        
            self.set_lp_data(i,_x,_g)
            self.cpx.solve()
            #print("status  sub   : {:>18s} {:>18.2f}".format(self.cpx.solution.status[self.cpx.solution.get_status()],self.cpx.solution.get_objective_value()))
            assert self.cpx.solution.get_status() == self.cpx.solution.status.optimal, "failed to solve subproblem to optimality"

            #s = self.cpx.solution.get_values(self.s)
            #_u = self.cpx.solution.get_values(self.u)
            #_v = np.array(self.cpx.solution.get_values(self.v))
            #s = min( np.min( (_u + _v) / self.data.pi[i] ), 1)            
            
            #error = _g[i] * s + self.cpx.solution.get_objective_value()      
            error = self.cpx.solution.get_objective_value()
            if error > 1e-6:
               is_cut = True       
               #_r = np.array(self.cpx.solution.get_values(self.r))
               _s = self.cpx.solution.get_values(self.s)
               _u = self.cpx.solution.get_values(self.u)
               _v = np.array(self.cpx.solution.get_values(self.v))
               
               #sumv = np.sum(_v)
               #sumr = np.sum(_r)
               #rhs = -(_u * self.data.gamma[i] + sumr)
               rhs = -(_u * self.data.gamma[i])
               self.cutinds[0] = self.data.g[i] 
               self.cutcoefs[0] = -_s
               self.cutcoefs[1:] = _v
               
               cuts.append(SparsePair(ind=self.cutinds.tolist(),val=self.cutcoefs.tolist()))
               rhss.append(rhs)
               
        return is_cut,cuts,rhss                          
            
    def set_lp_data(self,i,_x,_g):
        nj = self.data.nj
        J = range(nj)
        
        self.cpx.objective.set_linear([(self.s, _g[i])])                 
        self.cpx.objective.set_linear([(self.u, -self.data.gamma[i])])                 
        self.cpx.objective.set_linear([(self.v[j], -_x[j]) for j in J]) 
        #self.cpx.objective.set_linear([(self.v[j], -(_g[i]+_x[j])) for j in J]) 
        
        #my_coeffs = [(self.cnstr[j],self.v[k],-self.data.pi[i][j]) if k != j else (self.cnstr[j],self.v[k],-(self.data.pi[i][j]+1)) for k in J for j in J]
        #self.cpx.linear_constraints.set_coefficients(my_coeffs)
        my_coeffs = [(self.cnstr[j],self.s,self.data.pi[i][j]) for j in J]
        self.cpx.linear_constraints.set_coefficients(my_coeffs)
        
        #rhs = [(self.cnstr[j],-self.data.pi[i][j]) for j in J]
        #self.cpx.linear_constraints.set_rhs(rhs)
           
        self.cpx.set_problem_type(self.cpx.problem_type.LP)    
        
    def separate_oa_cuts(self,_g,_z):
        I = range(self.data.ni)
        is_cut = False
        cuts = []
        rhss = []
        total_phi = 0
        for i in I:
           phi = self.data.b[i]/(_g[i] + 1)
           grad = - self.data.b[i]/(_g[i] + 1)**2
           rhs = 1.000 * (phi - grad * _g[i])               
           #rhs = 1.0005 * (phi - grad * _g[i])
           total_phi += phi
           if (_z == None) or (phi - _z[i]) > 1e-4:
             is_cut = True
             cuts.append(SparsePair(ind=[self.data.z[i],self.data.g[i]],val=[1.0,-grad]))          
             rhss.append(rhs)
              
        return is_cut,cuts,rhss,total_phi    
          
    def separate_bd_cuts_analytically(self,_x,_g):        
        I,J = range(self.data.ni),range(self.data.nj)

        cuts = []
        rhss = []
        is_cut = False
        for i in I:
            csum = np.cumsum(_x[self.data.sorted_idpi[i]])
            gamma = self.data.gamma[i]
            g = _g[i]
            critical_k = np.where(csum > gamma)
            
            if len(critical_k[0]) > 0:
               cp = critical_k[0][0]
               cj = self.data.sorted_idpi[i][cp]            
               u = self.data.pi[i][cj]
            else:
               cp = -1
               u = 0.0   
            
            s = 1
            rhs = gamma * u
            
            of = g * s - rhs
            for j in J:
               sj = self.data.sorted_idpi[i][j]
               pi = self.data.pi[i][sj]
               xval = _x[sj]

               if cp == -1:
                  v = pi
               else:
                  if j < cp:
                     v = pi  - u
                  else:
                     v = 0.0 

               self.cutcoefs[sj+1] = v
               of -= xval * v    

            if of > 1e-6:
               is_cut = True
               self.cutcoefs[0] = -s
               self.cutinds[0] = self.data.g[i] 
               
               cuts.append(SparsePair(ind=self.cutinds.tolist(),val=self.cutcoefs.tolist()))
               rhss.append(-rhs)

        return is_cut,cuts,rhss    
        
def solve_callback(data,sub,ttime):
    I = range(data.ni)
    J = range(data.nj)
    
    cpx = data.cpx
    cpx.set_problem_type(cpx.problem_type.MILP)
    cpx.parameters.timelimit.set(ttime)
    #cpx.parameters.advance.set(2)
    #cpx.parameters.preprocessing.presolve.set(1)
    #cpx.parameters.preprocessing.symmetry.set(0)
    #cpx.parameters.preprocessing.reformulations.set(cpx.parameters.preprocessing.reformulations.values.none)
    #cpx.parameters.preprocessing.reformulations.set(cpx.parameters.preprocessing.reformulations.values.interfere_uncrush)
    #cpx.parameters.mip.tolerances.mipgap.set(1e-8)
    #cpx.parameters.mip.tolerances.mipgap.set(1e-8)
    cpx.parameters.mip.tolerances.integrality.set(1e-9)
    #cpx.parameters.mip.strategy.search.set(cpx.parameters.mip.strategy.search.values.traditional)
    cb = BDCutsCallback(data,sub)

    contextmask = cplex.callbacks.Context.id.candidate
    #contextmask |= cplex.callbacks.Context.id.relaxation
    #contextmask = cplex.callbacks.Context.id.thread_up
    #contextmask |= cplex.callbacks.Context.id.thread_down
    cpx.set_callback(cb, contextmask)

    startt = time()
    cpx.solve()
    rtime = time() - startt
    instance = str(data.ni) + "x" + str(data.nj) + "g" + str(data.gamma[0]) + "m" + str(data.u_scale_factor)
    print("Instancia  : ", instance) 
    #print("datafile   :{:18s}".format(data.filename))
    print("Objective  : {:18.2f}".format(cpx.solution.get_objective_value()))
    print("Status     : {:>18s} ".format(cpx.solution.status[cpx.solution.get_status()]))
    print("RMIPgap (%): {:18.8f}".format(100.0*cpx.solution.MIP.get_mip_relative_gap()))
    print("# bb nodes : {:18d}".format(cpx.solution.progress.get_num_nodes_processed()))
    print("Time (s)   : {:18.2f}".format(rtime))
    print()
    #_x = np.array(cpx.solution.get_values(data.x))
    #print(_x)
    return instance,cpx.solution.get_objective_value(),100*cpx.solution.MIP.get_mip_relative_gap(),cpx.solution.progress.get_num_nodes_processed(),rtime
             
def hot_start(data,sub,nhotstartloops,startt):
    I = range(data.ni)
    sub = Subproblem(data)
    
    cpx = cplex.Cplex(data.cpx)
    cpx.parameters.threads.set(1)
    cpx.parameters.lpmethod.set(2)
    #cpx.parameters.mip.tolerances.mipgap.set(1e-3)
    #cpx.parameters.mip.tolerances.integrality.set(1e-9)
    #cpx.parameters.timelimit.set(7200.0)
    #cpx.parameters.simplex.tolerances.markowitz.set(1e-1)
    #print(cpx.parameters.simplex.tolerances.markowitz.default())

    cpx.set_results_stream(None)
    cpx.set_log_stream(None)
    cpx.set_warning_stream(None)
    cpx.set_problem_type(cpx.problem_type.LP)        
    
    stop = False
    ub = float("inf")
    sup = float('inf')
    lb = 0
    it = 0
    gap = 100
    
    
    globalcuts = []
    globalrhss = []
    rtime = time() - startt             
    while (stop != True):
        it += 1
        run_normal = True
        try:
           cpx.solve()
       
        except CplexSolverError as exc:
           if exc.args[2] == cplex.exceptions.error_codes.CPXERR_NO_SOLN:
              print("Error solving OA MP in the warm start loop")
              raise
        
        #if cpx.get_problem_type() == cpx.problem_type.LP:
        #   assert cpx.solution.get_status() == cpx.solution.status.optimal, "Failed to solve problem to optimality"        
        #else:   
        #   assert cpx.solution.get_status() == cpx.solution.status.MIP_optimal or cpx.solution.get_status() == cpx.solution.status.optimal_tolerance, "Failed to solve problem to optimality"

        lb = cpx.solution.get_objective_value()
            
        _z = cpx.solution.get_values(data.z)
        _g = cpx.solution.get_values(data.g)
        _x = np.array(cpx.solution.get_values(data.x))
            
        is_cut1, cuts, rhss, total_phi = sub.separate_oa_cuts(_g,_z)
        if is_cut1 == True:
           cpx.linear_constraints.add(lin_expr=cuts,senses = ["G"]*len(cuts),rhs = rhss)        
           globalrhss.append(rhss)        
           globalcuts.append(cuts)
           
        '''   
        is_cut, cuts, rhss = sub.separate_bd_cuts_via_lp(_x,_g)
        if is_cut == True:
           cpx.linear_constraints.add(lin_expr=cuts,senses = ["G"]*len(cuts),rhs = rhss)        
           globalrhss.append(rhss)        
           globalcuts.append(cuts)           
        '''
            
        #'''
        is_cut, cuts, rhss = sub.separate_bd_cuts_analytically(_x,_g)
        if is_cut == True:
           cpx.linear_constraints.add(lin_expr=cuts,senses = ["G"]*len(cuts),rhs = rhss)        
           globalrhss.append(rhss)        
           globalcuts.append(cuts)           
        #'''
            
        if is_cut == False:
           sup = np.dot(data.f,_x) + np.sum([data.b[i]*(1/(_g[i] + 1)) for i in I])
        else:
           sup = float('inf')
            
        ub = min(ub,sup)
        if lb > 1e-6:
           gap = 100 * (ub - lb) / ub
           
        rtime = time() - startt             
        
        if it == nhotstartloops or gap < 1e-6 or rtime > maximumrunningtime:       
           stop = True
           
            
        
        print("{:5d}".format(it),end='')
        print(" {:18.2f}".format(lb),end='')
        print(" {:18.2f}".format(ub),end='')
        print(" {:18.8f} %".format(gap),end='')
        print(" {:18.2f} s".format(rtime),end='')
        print()
        #print(_x)
        
            
    for lhs,rhs in zip(globalcuts,globalrhss):
      #data.cpx.linear_constraints.advanced.add_lazy_constraints(lin_expr=lhs, senses=["G"]*len(rhs), rhs=rhs)    
      data.cpx.linear_constraints.add(lin_expr=lhs, senses=["G"]*len(rhs), rhs=rhs)    
       
    print("h{:5d}".format(it),end='')
    print(" {:18.2f}".format(lb),end='')
    print(" {:18.2f}".format(ub),end='')
    print(" {:18.8f}".format(gap),end='')
    print(" {:18.8f}".format(rtime),end='')
    print(" {:18b}".format(run_normal),end='')
    print(" {:18b}".format(len(globalcuts)),end='')
    print()
    return rtime,_x        
    #cpx.write("oamp.lp")      
            
def add_oa_initial_cuts(data,sub):
    I = range(data.ni)
    cpx = data.cpx
    _z = [-float('inf')  for i in I]
    for h in range(data.nprecuts):
        _g = [(np.sum( data.pi[ data.sorted_idpi[i][:min(data.gamma[i]+1,data.nj)] ] ) / data.nprecuts) * h  for i in I]
        
        is_feasible,cuts,rhss,total_phi = sub.separate_oa_cuts(_g,_z)             
        
        cpx.linear_constraints.add(lin_expr=cuts,senses = ["G"]*len(cuts),rhs = rhss)
            
def create_model(data):
   
    ni,nj = data.ni,data.nj
    I,J = range(ni),range(nj)
    
    cpx = cplex.Cplex()
    
    cpx.parameters.threads.set(1)
    cpx.parameters.timelimit.set(7200.00)
    cpx.parameters.mip.tolerances.mipgap.set(1e-5)
    cpx.parameters.mip.tolerances.integrality.set(1e-9)
    cpx.parameters.preprocessing.presolve.set(1)
    cpx.set_results_stream(None)
    cpx.set_log_stream(None)
    cpx.set_warning_stream(None)
    
    #cpx.parameters.mip.tolerances.mipgap.set(1e-8)
    #cpx.parameters.timelimit.set(3600.0)
    #cpx.parameters.timelimit.set(600)
    
    x = list(cpx.variables.add(obj = data.f.tolist(),lb = [0.0] * nj, ub = [1.0] * nj, types=['B'] * nj, names=['x('+str(j)+')' for j in J]))
    z = list(cpx.variables.add(obj = [1.0] * ni, lb=[0.0] * ni, ub = [cplex.infinity] * ni, types=['C'] * ni,names=['z('+str(i)+')' for i in I]))
    g = list(cpx.variables.add(obj = [0.0] * ni, lb=[0.0] * ni, ub = [cplex.infinity]* ni, types=['C'] * ni,names=['g('+str(i)+')' for i in I]))
    cpx.objective.set_sense(cpx.objective.sense.minimize)
                    
    #print('writing lp file...')           
    #cpx.write("oamp.lp")
    data.cpx = cpx
    data.x = x
    data.z = z
    data.g = g

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
            self.f = np.array(dt["f"])
            self.b = np.array(dt['b'])
            self.gamma = np.array(dt["gamma"])
            self.pi = np.array(dt['pi'])
            self.sorted_idpi = np.fliplr(np.argsort(self.pi,axis=1))
            self.sorted_pi = np.fliplr(np.sort(self.pi,axis=1))

            dist = cdist(self.pi.T,self.pi.T)
            self.neighbor = np.argsort(dist,axis=1)

class CData():
   def __init__(self,number_of_customers, number_of_facilities, u_scale_factor=1e5, gamma=5, is_uniform_gamma=True):
       np.random.seed(1)

       demand = np.random.randint(10, 1000,number_of_customers)
       location_customer_xy = np.random.uniform(0,1000,(number_of_customers,2))
       facility_location_xy = np.random.uniform(0,1000,(number_of_facilities,2))
       distance = cdist(location_customer_xy,facility_location_xy)
       facility_cost = np.full((number_of_facilities,),2000)
       #facility_cost = np.full((number_of_facilities,),50)

       u = np.rint(u_scale_factor/distance**2).astype(int)
       u0 = max(1, np.rint(u_scale_factor/50**2).astype(int))  
       #u = 1/distance**2
       #u0 = 1/50**2
       nzeros = np.count_nonzero(u)
       print(f"u non zero elements: {nzeros} / {u.size} ({100.0 * nzeros/u.size:6.2f} %) ")
       pi = u/u0

       self.ni = number_of_customers
       self.nj = number_of_facilities
       self.u_scale_factor = u_scale_factor
       I,J = range(self.ni),range(self.nj)

       if is_uniform_gamma == True:
          self.gamma = np.full((number_of_customers,),gamma)
       else:
          self.gamma = np.random.randint(1,gamma+1,number_of_customers)

       self.cxy = location_customer_xy
       self.fxy = facility_location_xy
       self.b = demand
       self.total_b = demand.sum()
       self.f = facility_cost 
       self.pi = pi
       self.u = u
       self.u0 = u0
       self.sorted_idpi = np.fliplr(np.argsort(pi,axis=1))
       self.sorted_pi = np.sort(pi,axis=1)[:,::-1] 
       self.u_scale_factor = u_scale_factor
       self.sorted_idu = np.fliplr(np.argsort(u,axis=1))
       self.sorted_u = np.sort(u,axis=1)[:,::-1] 
       dist = cdist(self.pi.T,self.pi.T)
       self.neighbor = np.argsort(dist,axis=1)
       self.sorted_inu = np.argsort(u,axis=1)       

'''                
if __name__ == "__main__":
   st = datetime.datetime.utcnow()-datetime.timedelta(hours=3)
   sts = time()
   print("starting time: ", st)
   #assert len(sys.argv) > 1,'please, provide a dat file'
   #filename = sys.argv[1]     
   #main(filename)
   main()
   et = datetime.datetime.utcnow()-datetime.timedelta(hours=3)    
   ets = time()
   print("Ending time: ", et)
   print("Total Time: ",et-st)
   print(f'Total Time: {ets-sts:.2f} s')
'''                
