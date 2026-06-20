import sys
import numpy as np
import cplex
from itertools import product
from time import perf_counter as time
from utils import *
'''
  paper   : An efficient branch-and-cut approach for large-scale competitive facility location problems with limited choice rule
  authors : Wei-Kun Chena, Wei-Yang Zhanga, Yan-Ru Wanga , Shahin Gelarehd, and Yu-Hong Dai
  date    : 9 Jun 2024
  url     : http://arxiv.org/abs/2406.05775v1

  abstract: We consider the competitive facility location problem with limited choice rule (CFLPLCR),
            which attempts to open a subset of facilities to maximize the net profit of a “newcomer” com-
            pany, requiring customers to patronize only a limited number of opening facilities and an outside
            option. We investigate the polyhedral structure of a mixed 0-1 set, defined by the function
            characterizing the probability of a customer patronizing the company’s open facilities, and
            propose an efficient branch-and-cut (B&C) approach for the CFLPLCR based on newly pro-
            posed mixed integer linear programming (MILP) formulations. Specifically, by establishing the
            submodularity of the probability function, we develop an MILP formulation for the CFLPLCR
            using the submodular inequalities. For the special case where each customer patronizes at
            most one open facility and the outside option, we show that the submodular inequalities can
            characterize the convex hull of the considered set and provide a compact MILP formulation.
            Moreover, for the general case, we strengthen the submodular inequalities by sequential lifting,
            resulting in a class of facet-defining inequalities. The proposed lifted submodular inequalities
            are shown to be stronger than the classic submodular inequalities, enabling to obtain another
            MILP formulation with a tighter linear programming (LP) relaxation. By extensive numerical
            experiments, we show that thanks to the tight LP relaxation, the proposed B&C approach out-
            performs the state-of-the-art generalized Benders decomposition approach by at least one order
            of magnitude. Furthermore, it enables to solve CFLPLCR instances with 10000 customers and
            2000 facilities.
'''
is_lifting = False
is_hotstart = True
is_cutloop = True
is_callback = True
lifting_running_time = 0.0

'''
  
  submodular main functions 
 
'''
def submodular_solve(data):
    global lifting_running_time
    lifting_running_time = 0.0
    mod = submodular_create_master_problem(data)
    run_time = 0.0
    sys.setrecursionlimit(10000)
    if is_cutloop == True:
       run_time = submodular_run_relaxed_cut_loop(data,mod,is_hotstart=is_hotstart)
    if is_callback == True or is_hotstart == True or is_cutloop == False:
       #is_lifting = True 
       submodular_solve_model_using_lazycallbacks(data,mod, run_time)
   
def submodular_solve_model_using_lazycallbacks(data,model,run_time=0.0,verbose=False):

    '''# setting parameters
    try:
       model.parameters.simplex.tolerances.feasibility = 1.00e-8
       model.parameters.mip.tolerances.mipgap = 1.00e-8
       #model.parameters.numericalemphasis = 1
    except Exception as e:
       print(f'could not set cplex parameters: {e}')
       sys.exit()'''

    model.parameters.simplex.tolerances.feasibility = 1.00e-8
    model.parameters.mip.tolerances.mipgap = 1.00e-8
    model.parameters.mip.display.set(2)
    model.parameters.threads.set(1)
    model.parameters.lpmethod.set(2)

    callback = LazyCallback(data,model)
    contextmask = cplex.callbacks.Context.id.candidate
    model.set_callback(callback, contextmask) 
    print("\n solving submodular using lazy callback")
    strt_time = time()
    status = model.solve()
    end_time = time()

    status_code = model.solution.get_status()
    status_string = model.solution.status[status_code]
    if status_code not in [model.solution.status.MIP_optimal, model.solution.status.optimal_tolerance, model.solution.status.MIP_abort_feasible]:
       print_error_msg(f"CPLEX could not solve the master problem using callbacks, status: {status_string}")

    objval = model.solution.get_objective_value()
    bbnodes = model.solution.progress.get_num_nodes_processed()
    print(status_string)
    print(f"market gain    : {objval:12.2f}")
    print(f"market lost    : {data.total_b - objval:12.2f}")
    print(f"total run time : {run_time + (end_time - strt_time):12.2f} s")
    print(f"# bb nodes     : {bbnodes:12.0f}")

'''
  
  lazy callback 
 
'''
class LazyCallback():    
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
       mod = self.mod
       I,J = range(dt.ni),range(dt.nj)
       
       _x = np.array(context.get_candidate_point(mod._x))
       _w = np.array(context.get_candidate_point(mod._w))
       _y = np.zeros(dt.nj+1).astype(int)

       is_cut_general = False
       for i in I:
           _, phi_s, coeffs14, coeffs15, is_cut = submodular_separate_submodular_cuts(dt,_x,_w,i,_y,is_lifting)

           if is_cut == True:
              if is_lifting == True: 
                 start = time() 
                 submodular_down_lifting_coefficients_cut14(dt,i,_x,_x,coeffs14,phi_s)
                 end = time()
                 global lifting_running_time 
                 lifting_running_time +=  end - start

                 start = time() 
                 submodular_up_lifting_coefficients_cut15(dt,i,_x,_x,coeffs15,phi_s)
                 end = time()
                 lifting_running_time +=  end - start

              submodular_add_cuts(dt,mod,i,_x,coeffs14,phi_s,context=context,is_callback=True)
              #print("----> (14) ---->")
              submodular_add_cuts(dt,mod,i,_x,coeffs15,phi_s,context=context,is_callback=True)
              #print("----> (15) ---->")

'''

  functions to lift submodular cuts

'''
def submodular_up_lifting_coefficients_cut15(dt,i,_x,_xr,coeffs,phi_s):
    Jl = range(dt.nj)
    # lifting sequence: NONDEACREASING order of relaxed x*
    L = sorted([(j,_xr[j]) for j in Jl],key=lambda t:t[1])
    # position of facility j in list L w.r.t. x*_j 
    pos = {j : posj for posj,(j,_) in enumerate(L)}
    # sorted list of js by x*_j
    J = [j for j,_ in L] 

    J = dt.sorted_inu[i]     


    LIFTED = [False]*dt.nj
    for l in J:
        if _x[l] < ZERO:
           P = set([j for j in J if (pos[j]<= pos[l] or _x[j] > 1 - ZERO) and LIFTED[j] == False])
           const = sum([coeffs[j] for j in P if _x[j] > 1 - ZERO and coeffs[j] <= ZERO]) + sum([coeffs[j] for j in P if pos[j] < pos[l] and _x[j] < ZERO and coeffs[j] <= ZERO])
           Q = list(set([j+1 for j in P if _x[j] > 1 - ZERO and coeffs[j] > ZERO]) | set([j+1 for j in P if pos[j] < pos[l] and _x[j] < ZERO and coeffs[j] > ZERO]))
           Q = [0] + Q 
           P = [int(j+1) for j in dt.sorted_idu[i] if pos[j] <= pos[l] or _x[j] > 1 - ZERO and j in P]
           PdiffQ = [int(j) for j in P if j not in Q]
           v_star_dp = submodular_get_v_max_star(dt,i,P,Q,PdiffQ,coeffs) - const
           v_star = v_star_dp           
           coeffs[l] = -1.0 * phi_s + sum([coeffs[j] for j in J if _x[j] > 1 - ZERO]) + v_star
           LIFTED[l] = True

def submodular_down_lifting_coefficients_cut14(dt,i,_x,_xr,coeffs,phi_s):
    Jl = range(dt.nj)
    # lifting sequence: NONDEACREASING order of relaxed x*
    L = sorted([(j,_xr[j]) for j in Jl],key=lambda t:t[1],reverse=True)
    # position of facility j in list L w.r.t. x*_j 
    pos = {j : posj for posj,(j,_) in enumerate(L)}
    # sorted list of js by x*_j
    J = [j for j,_ in L] 
    J = dt.sorted_inu[i]     
    LIFTED = [False]*dt.nj
    for l in J:        
        if _x[l] > 1 - ZERO:
           P = set([j for j in J if j != l and LIFTED[j] == False])
           const  = sum([coeffs[j] for j in P if _x[j] < ZERO and coeffs[j] <= ZERO]) + sum([coeffs[j] for j in P if pos[j] < pos[l] and _x[j] > 1 - ZERO and coeffs[j] <= ZERO])
           Q = list(set([j+1 for j in P if _x[j] < ZERO and coeffs[j] > ZERO]) | set([j+1 for j in P if pos[j] < pos[l] and _x[j] > 1 -  ZERO and coeffs[j] > ZERO]))
           Q = [0] + Q 
           P = [int(j+1) for j in dt.sorted_idu[i] if j != l and j in P]
           PdiffQ = [int(j) for j in P if j not in Q]
           v_star_dp = submodular_get_v_max_star(dt,i,P,Q,PdiffQ,coeffs) - const 
           v_star = v_star_dp
           coeffs[l] = phi_s - sum([coeffs[j] for j in J if pos[j] < pos[l] and _x[j] > 1 - ZERO]) - v_star
'''

  dynamic programming to solve the lifting problem

'''
def submodular_effe(dt,i,P,Q,PdiffQ,coeffs,tau):
    nq,np,npdiffq = len(Q),len(P),len(PdiffQ)
    return sum(dt.u[i][j-1] for j in PdiffQ[:min(tau, npdiffq)])

def submodular_Ze(dt,i,P,Q,PdiffQ,coeffs,t,llambda,tau):
    nq,np,npdiffq = len(Q),len(P),len(PdiffQ)
    if tau == 0:
        return llambda / (llambda + dt.u0)    
    if tau == t:
        sumu = sum(dt.u[i][j-1] for j in Q[1:t+1])
        suma = sum(coeffs[j-1] for j in Q[1:t+1])
        return -suma + ( (sumu + llambda) / (sumu + llambda + dt.u0) )
    
    jt = Q[t-1]
    return max(\
            submodular_Ze(dt,i,P,Q,PdiffQ,coeffs,t-1,llambda,tau),\
            -coeffs[jt-1] + submodular_Ze(dt,i,P,Q,PdiffQ,coeffs,t-1,llambda + dt.u[i][jt-1],tau-1) )         
    
def submodular_get_v_max_star(dt,i,P,Q,PdiffQ,coeffs):
    nq,np,npdiffq = len(Q),len(P),len(PdiffQ)
    v_star = 0.0
    Z = 0.0
    for tau in range(min((dt.gamma[i]+1),nq+1)):
        Z = max(v_star, submodular_Ze(dt,i,P,Q,PdiffQ,coeffs,nq,submodular_effe(dt,i,P,Q,PdiffQ,coeffs,dt.gamma[i]-tau),tau))
        if (Z>v_star):
            v_star = Z ;
    return v_star

'''

  functions used to separated submodular custs

'''
def submodular_separate_submodular_cuts(dt,_x,_w,i,_y,is_lifting=False):
    J = range(dt.nj)
    total_phi = 0.0
    is_cut = False

    phi_s, u_s, u_ks = submodular_compute_phi_s(dt,i,_x,_y)

    total_captured_demand = dt.b[i] * phi_s

    _x1  = np.full((dt.nj,),1)
    coeffs14 = np.zeros((dt.nj,))
    coeffs15 = np.zeros((dt.nj,))

    # violated cut? w > phi_s
    is_cut=False
    if _w[i] - phi_s >= 1.e-4 :
       is_cut=True
       for j in J:
           if _x[j] < ZERO:
              coeffs14[j] = submodular_compute_cut_coefficient_rho(dt,i,j,u_s,u_ks)
              coeffs15[j] = submodular_compute_cut_coefficient_rho(dt,i,j,0.0,0.0)
           else: 
              _x1[j] = 0 
              _, u_s1, u_ks1 = submodular_compute_phi_s(dt,i,_x1,_y)
              coeffs14[j] = submodular_compute_cut_coefficient_rho(dt,i,j,u_s1,u_ks1)
              _x1[j] = 1                                                 

              _x[j] = 0 
              _, u_s0, u_ks0 = submodular_compute_phi_s(dt,i,_x,_y)
              coeffs15[j] = submodular_compute_cut_coefficient_rho(dt,i,j,u_s0,u_ks0)
              _x[j] = 1
              
    return total_captured_demand, phi_s, coeffs14,coeffs15, is_cut

def submodular_compute_cut_coefficient_rho(dt,i,j,u_s,u_ks):
    # computing the cut coefficient for the j facility 
    u_j = dt.u[i][j]
    uj_uks = max(0.0, u_j - u_ks)
    num = dt.u0*uj_uks
    den = (u_s + uj_uks + dt.u0) * (u_s + dt.u0) 
    rho = num/den
    return rho

def submodular_compute_phi_s(dt,i,_x,_y):
    _y[0] = 0
    if np.all(_x == 0):
       return 0.0,0.0,0.0
    cumsum = dt.gamma[i]
    u_s,u_ks = 0.0,0.0
    h = 0
    jks,ks = None,None
    _y[0] = 0
    while cumsum > ZERO and h < dt.nj:
        j = dt.sorted_idu[i][h]
        val_x = _x[j]
        if val_x > ZERO:
           u = dt.u[i][j]
           _y[0] += 1
           _y[_y[0]] = j 
           if cumsum > val_x:
              cumsum -= val_x
           else:
              val_x = cumsum
              u_ks = dt.u[i][j]
              cumsum = 0.0
              jks,ks = j,h
           u_s += u * val_x
        h += 1
    phi_s = u_s/(u_s + dt.u0) 

    return phi_s,u_s,u_ks

def submodular_run_relaxed_cut_loop(dt,mod,is_hotstart=True,verbose=False):
    global lifting_running_time 
    '''
       hot start or warmstart cut loop 
    '''
    I, J = range(dt.ni), range(dt.nj)

    # ----------------------------------
    # relaxing the master problem
    # ----------------------------------
    if is_hotstart == True:
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
       mod._w = mip_mod._w

    print(f"\n\nSubmodular relaxed cutting loop: ",end="")
    if is_lifting == True:
       print( "lifted ",end="")
    if is_hotstart == True:
       print( "hotstart ",end="")
    print()

    best_x,best_s = None,None
    old_ub = np.inf
    _y = np.zeros(dt.nj+1).astype(int)

    stop,ub,lb,it,starting_time = False,float("inf"),-float("inf"),0,time()
    same_it = 0
    while (stop == False):
       it = it + 1

       # solving the master probem
       mod.solve()
       status_code = mod.solution.get_status()
       status_string = mod.solution.status[status_code]
       if is_hotstart == True:
          if status_code != mod.solution.status.optimal:
              print_error_msg(f"CPLEX could not solve the master problem using callbacks, status: {status_string}")
       else:
          if status_code not in [mod.solution.status.MIP_optimal, mod.solution.status.optimal_tolerance]:
             print_error_msg(f"CPLEX could not solve the master problem using callbacks, status: {status_string}")

       ub = mod.solution.get_objective_value()
       _xrelaxed = np.array(mod.solution.get_values(mod._x))
       _w        = np.array(mod.solution.get_values(mod._w))

       # -----------------------------------
       # heuristic to round master problem solutiion
       # sort x* by x*_j1 > ... x*_jn 
       # S = {j : x* = 1}
       # _S = {j : 0 < x* < 1 sorted by x*_j1 > ... > x*_jn}
       # for l in _S
       #     S = S U {l}
       #     separate the submodular cuts (14)-(15)
       #     down lifting
       #         P = {j in S : sorted by x* non-dereasing}
       #     up lifting 
       #         P = {j in S : sorted by x* non-increasing}
       # -----------------------------------
       S1 = np.where( _xrelaxed > 1.0-ZERO)[0]
       Sf = np.where( (ZERO < _xrelaxed) & (_xrelaxed < 1.0-ZERO))[0]
       sorted_Sf = sorted([ (j,_xrelaxed[j]) for j in Sf],key = lambda t:t[1],reverse=True)

       _x = np.zeros(dt.nj).astype(int)
       _x[S1] = 1

       # current S w/ no fractional xs
       sup = - (_x * dt.f).sum()
       for i in I:
           supi , phi_s, coeffs14, coeffs15, is_cut = submodular_separate_submodular_cuts(dt,_x,_w,i,_y,is_lifting)
       
           sup += supi
           if is_cut == True:
              if is_lifting == True: 
                 start = time()
                 submodular_down_lifting_coefficients_cut14(dt,i,_x,_xrelaxed,coeffs14,phi_s)
                 end = time()
                 global lifting_running_time 
                 lifting_running_time +=  end - start

                 start = time()
                 submodular_up_lifting_coefficients_cut15(dt,i,_x,_xrelaxed,coeffs15,phi_s)
                 end = time()
                 
                 lifting_running_time +=  end - start
              #w[i] <= phi_s + sum(coeffs[j] * (x[j] - _x[j]) for j in J)
              submodular_add_cuts(dt,mod,i,_x,coeffs14,phi_s)
              submodular_add_cuts(dt,mod,i,_x,coeffs15,phi_s)

       if sup > lb:
          lb = sup
          best_x = _x.copy()

       # current S w/ fractional xs, one at a time
       best_sup = sup 
       for j,_ in sorted_Sf:        
           _x[j] = 1
           
           # separating submodular cuts 
           supj = - (_x * dt.f).sum()
           is_next_j = False
           '''
           for i in I:
               supi , phi_s, coeffs14, coeffs15, is_cut = submodular_separate_submodular_cuts(dt,_x,_w,i,_y,is_lifting)
           
               supj += supi
               if is_cut == True:
                  is_next_j = True
                  if is_lifting == True: 
                     start = time() 
                     submodular_down_lifting_coefficients_cut14(dt,i,_x,_xrelaxed,coeffs14,phi_s)
                     end = time()
                     lifting_running_time +=  end - start

                     start = time() 
                     submodular_up_lifting_coefficients_cut15(dt,i,_x,_xrelaxed,coeffs15,phi_s)
                     end = time()
                     lifting_running_time +=  end - start

                  #w[i] <= phi_s + sum(coeffs[j] * (x[j] - _x[j]) for j in J)
                  submodular_add_cuts(dt,mod,i,_x,coeffs14,phi_s)
                  submodular_add_cuts(dt,mod,i,_x,coeffs15,phi_s)
           '''
           if supj > best_sup:
              best_sup = supj
              best_s = _x.copy()
           else:
              break
           if is_next_j == False:
              break
       sup = best_sup
       # updating the lower bound 
       if sup > lb:
          lb = sup
          best_x = best_s.copy()
        
       gap = 100.0 if ub < ZERO else 100.0 * (ub - lb)/ub

       # stop criterion
       if old_ub - ub < ZERO:
          same_it += 1
       if gap < 0.01 or same_it == 2:
          stop = True
       
       old_ub = ub

       # printing stats
       running_time = time() - starting_time
       if is_hotstart == True:
          print('h',end='')
       else:
          print('i',end='')
       print(f"{it:4d} {ub:10.2f} {lb:10.2f} {sup:10.2f} {gap:8.2f} %  {running_time:10.2f} s")

    # ----------------------------------
    # returning with the binary variables
    # and providing the first feasible solution
    # ----------------------------------
    if is_hotstart == True:
       n_constr_mip = mip_mod.linear_constraints.get_num()
       n_constr_lp = mod.linear_constraints.get_num()
       for r in range(n_constr_mip,n_constr_lp):
           lin_expr = mod.linear_constraints.get_rows(r)
           sense = mod.linear_constraints.get_senses(r)
           rhs = mod.linear_constraints.get_rhs(r)
           mip_mod.linear_constraints.add(lin_expr=[lin_expr],senses=[sense],rhs=[rhs])
       mod = mip_mod
       submodular_set_master_problem_initial_integer_feasible_solution(mip_mod,best_x)
    else:
       print(f"market gain : {lb:12.2f}")
       print(f"market lost : {dt.total_b - ub:12.2f}")
       print(f"run time    : {running_time:12.2f} s")
    return running_time

def submodular_set_master_problem_initial_integer_feasible_solution(mod,_x):
     dt = mod._dt
     J = range(dt.nj)
     mod.MIP_starts.add([(cplex.SparsePair(ind=[mod._x[j]],val=[int(_x[j])]),mod.MIP_starts.effort_level.repair) for j in J])

def submodular_add_cuts(dt,mod,i,_x,coeffs,phi_s,context=None,is_callback=False):
    #w[i] - sum(coeffs[j] x[j] for j in J)  <= phi_s - sum(coeffs[j]  _x[j]) for j in J)
    J = range(dt.nj)
    inds = [mod._w[i]] + [mod._x[j] for j in J]
    vals = [1.0] + [-float(coeffs[j]) for j in J]
    rhs =  phi_s - sum(int(_x[j]) * float(coeffs[j]) for j in J)
    cut = [cplex.SparsePair(ind=inds,val=vals)]
    if is_callback == False:
       mod.linear_constraints.add(lin_expr=cut,senses=["L"],rhs=[rhs])
    else:
       context.reject_candidate(constraints=cut,senses=["L"],rhs=[rhs]) 

def submodular_create_master_problem(dt):
    I, J      = range(dt.ni), range(dt.nj)
    mod  = cplex.Cplex()

    mod.parameters.threads.set(1)
    #mod.parameters.mip.tolerances.integrality.set(1e-9)
    mod.parameters.mip.display.set(2)
    #mod.set_results_stream(None)
    #mod.set_log_stream(None)
    #mod.set_warning_stream(None)


    mod._x = list(mod.variables.add(types=['B']*dt.nj, names=[f"x_{j}" for j in J]))
    mod._w = list(mod.variables.add(lb=[0.0]*dt.ni,ub=[1.0]*dt.ni,types=['C']*dt.ni, names=[f"w_{i}" for i in I]))
    
    mod.objective.set_linear( [ (mod._x[j],-dt.f[j]) for j in J ] + [(mod._w[i],dt.b[i]) for i in I]) 
      
    mod.objective.set_sense(mod.objective.sense.maximize)
    mod._dt = dt
    return mod 
