import numpy as np
from numpy import newaxis
from itertools import product
from math import ceil, floor
from time import perf_counter as time
from heur.heur import CRoundingHeur 
import cplex
from utils import *

def conic_solve(dt,verbose=False):
    I, J      = range(dt.ni), range(dt.nj)
    mod  = cplex.Cplex()
    print("\n solving conic via cplex")

    mod._dt = dt

    mod.parameters.mip.tolerances.integrality.set(1e-9)
    mod.parameters.mip.display.set(2)
    mod.parameters.threads.set(1)    
    #mod.set_results_stream(None)
    #mod.set_log_stream(None)
    #mod.set_warning_stream(None)

    mod.x = list(mod.variables.add(obj=dt.f,types=['B']*dt.nj, names=[f"x_{j}" for j in J]))
    mod.y = {(i,j) : list(mod.variables.add(types=['B'],names=[f'y_{i}_{j}']))[0]  for (i,j) in product(I,J)}
    mod.beta = list(mod.variables.add(obj=dt.b,lb=[0.0]*dt.ni,ub=[1.0]*dt.ni,types=['C']*dt.ni, names=[f"beta_{i}" for i in I]))
    mod.z = list(mod.variables.add(lb=[1.0]*dt.ni,types=['C']*dt.ni,names=[f'z_{i}' for i in I]))

    mod.objective.set_sense(mod.objective.sense.minimize)
 
    for (i,j) in product(I,J):
        inds = [mod.x[j],mod.y[i,j]]
        vals = [-1.0,1.0]
        rhs = 0.0
        cut = [cplex.SparsePair(ind=inds,val=vals)]
        mod.linear_constraints.add(lin_expr=cut,senses=['L'],rhs=[rhs])
       
    for i in I:
        inds = [mod.y[i,j] for j in J] + [mod.z[i]]
        vals = [dt.pi[i][j] for j in J] + [-1.0]
        rhs = -1
        cut = [cplex.SparsePair(ind=inds,val=vals)]
        mod.linear_constraints.add(lin_expr=cut,senses=['E'],rhs=[rhs])

        inds = [mod.y[i,j] for j in J]
        vals = [1.0] * dt.nj 
        rhs = float(dt.gamma[i])

        cut = [cplex.SparsePair(ind=inds,val=vals)]
        mod.linear_constraints.add(lin_expr=cut,senses=['L'],rhs=[rhs])

        inds1 = [mod.beta[i]]
        inds2 = [mod.z[i]]
        vals = [1.0]
        rhs = 1.0
        cut = cplex.SparseTriple(ind1=inds1,ind2=inds2,val=vals) 
        mod.quadratic_constraints.add(quad_expr=cut, sense="G", rhs=rhs)

    start_time = time()
    status = mod.solve()
    end_time = time()
    status_code = mod.solution.get_status()
    status_string = mod.solution.status[status_code]
    if status_code not in [mod.solution.status.MIP_optimal, mod.solution.status.optimal_tolerance]:
       print_error_msg(f"CPLEX could not solve the master problem using callbacks, status: {status_string}")

    objval = mod.solution.get_objective_value()
    bbnodes = mod.solution.progress.get_num_nodes_processed()
    _x = np.array(mod.solution.get_values(mod.x))

    print(status_string)
    print(f"market gain    : {dt.total_b - objval:12.2f}")
    print(f"market lost    : {objval:12.2f}")
    print(f"total run time : {end_time - start_time:12.2f} s")
    print(f"# bb nodes     : {bbnodes:12.0f}")
    print(f"# Opt. struct  : ")
    for j in J:
        if _x[j] > 0.9:
            print(j, " : ", _x[j])
    print() 
