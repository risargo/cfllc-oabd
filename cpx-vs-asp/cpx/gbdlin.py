import numpy as np
from itertools import product
from time import perf_counter as time
import cplex
from cplex.exceptions import CplexError

from math import ceil, floor
from time import perf_counter as time
from utils import *
'''
  paper   : Branch-and-cut approach based on generalized benders decomposition for facility location with limited choice rule
  authors : Yun Hui Lin, Qingyun Tian 
  date    : 16 December 2020
  url     : https://doi.org/10.1016/j.ejor.2020.12.017

  abstract: This paper studies the exact solution approaches for a generalized competitive facility location problem.
            We consider a company that plans to introduce a service by opening a set of facilities. The objective
            is to maximize the proﬁt taking into account the revenue and the ﬁxed cost. It is assumed that when
            customers are offered with a set of open facilities, they ﬁrst form the consideration set, i.e., the subset
            of open facilities that the customers are willing to patronize. They then split the buying power among
            the facilities in the set plus some outside option, according to Luce’s choice axiom. The resulting location
            problem provides a generalized framework that covers many existing models in competitive facility loca-
            tion problems where customers follow either the proportional choice rule or the partially binary choice
            rule. As our main contribution, we propose a branch-and-cut algorithm based on the generalized Ben-
            ders decomposition scheme (B&C-Benders), which projects out high-dimensional continuous variables in
            modeling the consideration set and only works on the projected decision space. Our extensive computa-
            tional experiment shows that B&C-Benders outperforms state-of-the-art exact approaches, both in terms
            of the computational time, and in terms of the number of instances solved to optimality. In the special
            case where customers follow the partially binary choice rule, B&C-Benders turns out to be eﬃcient for
            large-scale instances with thousands of customer zones and hundreds of facilities.
'''
class GBDLinLazyCutCallback():
    def __init__(self,data,mod):
      self.data = data
      self.mod = mod
      self.cutcoeffs = np.zeros(data.nj+1).astype(float)
      self.cutinds = np.zeros(data.nj+1).astype(int)
      self.cutinds[1:] = mod._x

      self.nogoodcoeffs = np.zeros(data.ni + data.nj).astype(float)      
      self.nogoodinds = np.zeros(data.ni + data.nj).astype(int)
      self.nogoodinds[mod._z] = mod._z
      self.nogoodinds[mod._x] = mod._x
      self.nogoodcoeffs[mod._z] = 1.0
      self.Jtilde = [0.0] * data.nj 
    def invoke(self, context):
        try:
           if context.in_candidate():
              self.separate_cuts(context)
        except:
           info = sys.exc_info()
           print('#### Exception in callback: ', info[0])
           print('####                        ', info[1])
           print('####                        ', info[2])
           traceback.print_tb(info[2], file=sys.stdout)
           raise 
    def separate_cuts(self,context):
        dt = self.data
        Jtilde = self.Jtilde
        I,J = range(dt.ni),range(dt.nj)
        mod = self.mod
        _x = np.array(context.get_candidate_point(mod._x))
        _z = np.array(context.get_candidate_point(mod._z))
        
        is_feasible = True
        total_phi = 0.0
        for i in I:
            _g = 0.0
            gamma = dt.gamma[i]
            h = 0
            for j in dt.sorted_idpi[i]:
                if _x[j] > 1 - ZERO:
                   Jtilde[h] = j
                   h += 1
 
                   _g += dt.pi[i][j]
                   gamma -= 1
                   if gamma == 0:
                      break  

            _phi = dt.b[i]/(_g + 1)         

            total_phi += _phi     

            #if abs(_z[i] - _phi) > ZERO:
            if abs(_z[i] - _phi) > 1e-2:
               is_feasible = False
               if h < gamma: 
                  _ll = [(dt.b[i]*dt.pi[i][j])/(_g + 1)**2 for j in J] 
               else: 
                  J0 = [j for j in dt.sorted_idpi[i] if j not in Jtilde[:h] and _x[j] > 1 - ZERO]
                  pihat = 0.0 if len(J0) == 0 else dt.pi[i][J0[0]] 
                  _ll = [0.0 if j in J0 else (dt.b[i] * max(dt.pi[i][j] - pihat,0.0))/(_g + 1)**2 for j in J] 
               
               rhs = _phi + np.dot(_ll,_x)

               self.cutcoeffs[0] = 1.0
               self.cutinds[0] = mod._z[i]
               self.cutcoeffs[1:] = _ll

               cut = [cplex.SparsePair(ind=self.cutinds.tolist(),val=self.cutcoeffs.tolist())]
               context.reject_candidate(constraints=cut,senses=["G"],rhs=[rhs]) 
        '''
        if is_feasible == False:
           rhs = total_phi 
           for j in J:
               if _x[j] > 1 - ZERO:
                  #self.nogoodcoeffs[j] = -total_phi
                  #rhs -= total_phi 
                  self.nogoodcoeffs[j] = 0.0
               else:
                  self.nogoodcoeffs[j] = total_phi
            
           cut = [cplex.SparsePair(ind=self.nogoodinds.tolist(),val=self.nogoodcoeffs.tolist())]
           context.reject_candidate(constraints=cut,senses=["G"],rhs=[rhs]) 
        '''

def gbdlin_solve(dt,verbose=False):
    print("\n solving gbd lin using cpx lazy callback")
    mod = gbdlin_create_model(dt)
    mod.parameters.threads.set(1)
    mod.parameters.mip.display.set(2)
    callback = GBDLinLazyCutCallback(dt,mod)
    contextmask = cplex.callbacks.Context.id.candidate
    mod.set_callback(callback, contextmask) 
    startt = time()
    status = mod.solve()
    end_time = time()

    status_code = mod.solution.get_status()
    status_string = mod.solution.status[status_code]
    if status_code not in [mod.solution.status.MIP_optimal, mod.solution.status.optimal_tolerance]:
       print_error_msg(f"CPLEX could not solve the gbd lin master problem using callbacks, status: {status_string}")

    objval = mod.solution.get_objective_value()
    bbnodes = mod.solution.progress.get_num_nodes_processed()
    print(f"market gain    : {dt.total_b - objval:12.2f}")
    print(f"market lost    : {objval:12.2f}")
    print(f"total run time : {end_time - startt:12.2f} s")
    print(f"# bb nodes     : {bbnodes:12.0f}")
    print()
    _x = np.array(mod.solution.get_values(mod._x))
    print() 
    print(*np.where(_x>0.9)[0].tolist())
    print() 

def gbdlin_create_model(dt):
    ni,nj = dt.ni,dt.nj
    I,J = range(ni),range(nj)
    mod  = cplex.Cplex()

    mod.parameters.threads.set(1)
    mod.parameters.mip.tolerances.integrality.set(1e-9)
    #mod.set_results_stream(None)
    #mod.set_log_stream(None)
    #mod.set_warning_stream(None)

    mod._x = list(mod.variables.add(obj=dt.f,lb=[0.0]*dt.nj,ub=[1.0]*dt.nj,types=['B']*dt.nj, names=[f"x_{j}" for j in J]))
    mod._z = list(mod.variables.add(obj=[1.0]*dt.ni,lb=[0.0]*dt.ni,types=['C']*dt.ni, names=[f"z_{i}" for i in I]))
    mod.objective.set_sense(mod.objective.sense.minimize)

    mod._dt = dt
    return mod
