import numpy as np
from numpy import newaxis
from itertools import product
from math import ceil, floor
from time import perf_counter as time
from heur.heur import CRoundingHeur 
import cplex
from utils import *
'''
  oabd method
 
'''
cutcoeffs = [] 
cutinds = [] 
 
def oabd_solve(data,verbose=False):
    print("\n solving oabd via cplex")
    
    mod = oabd_create_model(data,verbose)

    global cutcoeffs
    cutcoeffs = [0.0] * (data.nj+1)
    global cutinds
    cutinds = [0] + mod._x

    startt = time()

    oabd_add_oa_initial_cuts(data,mod)

    _x = oabd_hotstart(data,mod,startt,is_oabd_hotstart=True)

    '''
    stheur = time()
    heur = CRoundingHeur(data,_x)
    etheur = time()
    print(f'heur time: {etheur-stheur:.2f} s')
    print(f'heur obj : {heur.get_obj():18.2f} ')
    print()
    oabd_add_heur_cuts(data,mod,heur.x,heur.z)

    set_master_problem_initial_integer_feasible_solution(mod,heur.x,heur.z)
    '''

    callback = BDCutsCallback(data,mod)
    contextmask = cplex.callbacks.Context.id.candidate
    mod.set_callback(callback, contextmask) 
    mod.parameters.mip.display.set(2)

    strt_time = time()
    status = mod.solve()
    end_time = time()
    status_code = mod.solution.get_status()
    status_string = mod.solution.status[status_code]
    if status_code not in [mod.solution.status.MIP_optimal, mod.solution.status.optimal_tolerance]:
       print_error_msg(f"CPLEX could not solve the master problem using callbacks, status: {status_string}")

    objval = mod.solution.get_objective_value()
    bbnodes = mod.solution.progress.get_num_nodes_processed()
    print(status_string)
    print(f"market gain    : {data.total_b - objval:12.2f}")
    print(f"market lost    : {objval:12.2f}")
    print(f"total run time : {end_time - startt:12.2f} s")
    print(f"# bb nodes     : {bbnodes:12.0f}")

def oabd_create_model(data,verbose=False):
    ni,nj = data.ni,data.nj
    I,J = range(ni),range(nj)
    mod  = cplex.Cplex()

    mod.parameters.threads.set(1)
    #mod.parameters.mip.tolerances.integrality.set(1e-9)
    #mod.set_results_stream(None)
    #mod.set_log_stream(None)
    #mod.set_warning_stream(None)

    mod._x = list(mod.variables.add(obj=data.f,lb=[0.0]*data.nj,ub=[1.0]*data.nj,types=['B']*data.nj, names=[f"x_{j}" for j in J]))
    mod._z = list(mod.variables.add(obj=[0.0]*data.ni,lb=[0.0]*data.ni,types=['C']*data.ni, names=[f"z_{i}" for i in I]))
    mod._g = list(mod.variables.add(obj=[1.0]*data.ni,lb=[0.0]*data.ni,types=['C']*data.ni, names=[f"g_{i}" for i in I]))

    mod.objective.set_sense(mod.objective.sense.minimize)
                   
    mod._dt = data
    return mod    

def oabd_add_oa_initial_cuts(data,mod):
    I = range(data.ni)
    nprecuts = data.npc #int(data.nj/10)
    for h in range(nprecuts):
        for i in I:
            _z = (np.sum( data.pi[ data.sorted_idpi[i][:min(data.gamma[i]+1,data.nj)] ] ) / nprecuts) * h 
            _g = 0.0
            oabd_add_oa_cuts(data,mod,i,_z,_g) 

def oabd_add_oa_cuts(data,mod,i,_z,_g,context=None,is_callback=False):
    is_cut = False
    _phi = data.b[i]/(_z+1)
    _delphi = - data.b[i]/(_z+1)**2
    rhs = _phi - _delphi * _z 
    #if _phi - _g > 1e-4:
    if 100 * ( max(0, ( _phi - _g )) /_phi ) > 1.00e-2: 
       cut = [cplex.SparsePair(ind=[mod._g[i],mod._z[i]],val=[1.0,-_delphi])]
       is_cut = True  
       if is_callback == False:
          mod.linear_constraints.add(lin_expr=cut,senses=["G"],rhs=[rhs])
       else:
          context.reject_candidate(constraints=cut,senses=["G"],rhs=[rhs]) 
    return is_cut 

def oabd_add_bd_cuts(data,mod,i,_x,_z,context=None,is_callback=False):    
    is_cut = False
    J = range(data.nj)
    global cutcoeffs
    global cutinds

    csum = np.cumsum(_x[data.sorted_idpi[i]])
    gamma = data.gamma[i]
    critical_k = np.where(csum > gamma)

    if len(critical_k[0]) > 0:
       cp = critical_k[0][0]
       cj = data.sorted_idpi[i][cp]            
       u = data.pi[i][cj]
    else:
       cp = -1
       u = 0.0   
    
    s = 1
    rhs = gamma * u
    
    of = _z * s - rhs
    for j in J:
       sj = data.sorted_idpi[i][j]
       pi = data.pi[i][sj]
       xval = _x[sj]

       if cp == -1:
          v = pi
       else:
          if j < cp:
             v = pi  - u
          else:
             v = 0.0 

       cutcoeffs[sj+1] = v
       of -= xval * v    

    if of > 1.e-6:
       cutcoeffs[0] = -s
       cutinds[0] = mod._z[i]

       cut = [cplex.SparsePair(ind=cutinds,val=cutcoeffs)]
       is_cut = True
       if is_callback == False:
          mod.linear_constraints.add(lin_expr=cut,senses=["G"],rhs=[-rhs])
       else:
          context.reject_candidate(constraints=cut,senses=["G"],rhs=[-rhs]) 
    return is_cut

def oabd_add_heur_cuts(data,mod,_x,_z):
    I = range(data.ni)
    for i in I:
        oabd_add_oa_cuts(data,mod,i,_z[i],0.0) 
        is_cut = oabd_add_bd_cuts(data,mod,i,_x,_z[i])

def oabd_hotstart(data,mod,start_time,is_oabd_hotstart=False):
    I,J = range(data.ni),range(data.nj)
    nhotstartloops = 20 #int(data.ni/10)

    # relax master problem
    if is_oabd_hotstart == True:
       mip_mod = mod
       mod = cplex.Cplex(mod)
       mod.parameters.threads.set(1)
       mod.parameters.lpmethod.set(2)
       #mod.parameters.simplex.tolerances.markowitz.set(1e-4)
       mod.set_results_stream(None)
       mod.set_log_stream(None)
       mod.set_warning_stream(None)
       mod.set_problem_type(mod.problem_type.LP) 
       mod._x = mip_mod._x
       mod._z = mip_mod._z
       mod._g = mip_mod._g

    stop = False
    ub,sup = float("inf"),float('inf')
    lb,it,gap = 0,0,100.0
    rtime = time() - start_time             
    while (stop == False):
       it += 1
       mod.solve()
       status_code = mod.solution.get_status()
       status_string = mod.solution.status[status_code]
       if is_oabd_hotstart == True:
          if status_code != mod.solution.status.optimal:
              print_error_msg(f"CPLEX could not solve the master problem using callbacks, status: {status_string}")
       else:
          if status_code not in [mod.solution.status.MIP_optimal, mod.solution.status.optimal_tolerance]:
             print_error_msg(f"CPLEX could not solve the master problem using callbacks, status: {status_string}")

       lb = mod.solution.get_objective_value()
       _x = np.array(mod.solution.get_values(mod._x))
       _z = mod.solution.get_values(mod._z)
       _g = mod.solution.get_values(mod._g)

       is_global_cut = False
       for i in I:
           oabd_add_oa_cuts(data,mod,i,_z[i],_g[i]) 
           is_cut = oabd_add_bd_cuts(data,mod,i,_x,_z[i])
           if is_cut == True:
              is_global_cut = True

       if is_global_cut == False:
          sup = np.dot(data.f,_x) + np.sum([data.b[i]/(_z[i]+1) for i in I])
       else:
          sup = float('inf')
           
       ub = min(ub,sup)
       if lb > ZERO:
          gap = 100 * (ub - lb) / ub
          
       rtime = time() - start_time            
    
       if (is_oabd_hotstart == True and it == nhotstartloops) or gap < 1.e-2:
          stop = True

       if is_oabd_hotstart == True:
          print("h",end='')
       else:
          print("i",end='')

       print(" {:5d}".format(it),end='')
       print(" {:18.2f}".format(lb),end='')
       print(" {:18.2f}".format(ub),end='')
       print(" {:18.8f} %".format(gap),end='')
       print(" {:18.2f} s".format(rtime),end='')
       print()
    # end while    
    if is_oabd_hotstart == True:
       n_constr_mip = mip_mod.linear_constraints.get_num()
       n_constr_lp = mod.linear_constraints.get_num()
       for r in range(n_constr_mip,n_constr_lp):
           lin_expr = mod.linear_constraints.get_rows(r)
           sense = mod.linear_constraints.get_senses(r)
           rhs = mod.linear_constraints.get_rhs(r)
           mip_mod.linear_constraints.add(lin_expr=[lin_expr],senses=[sense],rhs=[rhs])
       mod = mip_mod
    return _x

def set_master_problem_initial_integer_feasible_solution(mod,_x,_z):
    
    newsol = mod.MIP_starts.add(\
            cplex.SparsePair(ind= mod._x + mod._z, val= _x.tolist() + _z.tolist()),\
            mod.MIP_starts.effort_level.auto)

class BDCutsCallback():    
    def __init__(self,data,mod):
      self.data = data
      self.mod = mod
      
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
      I = range(dt.ni)

      mod = self.mod
      
      _x = np.array(context.get_candidate_point(mod._x))
      _g = context.get_candidate_point(mod._g)
      _z = context.get_candidate_point(mod._z)      
     
      for i in I:
          oabd_add_oa_cuts(dt,mod,i,_z[i],_g[i],context=context,is_callback=True) 
          oabd_add_bd_cuts(dt,mod,i,_x,_z[i],context=context,is_callback=True)
      
