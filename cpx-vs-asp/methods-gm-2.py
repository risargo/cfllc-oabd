from data.data import CData 
# gurobi
#from grb.conic import conic_solve as grb_conic_solve
#from grb.gbdlin import gbdlin_solve as grb_gbdlin_solve
#from grb.submodular import submodular_solve as grb_submodular_solve
#from grb.oabd import oabd_solve as grb_oabd_solve 

# docplex
#from docpx.conic import conic_solve as docpx_conic_solve
#from docpx.gbdlin import gbdlin_solve as docpx_gbdlin_solve
#from docpx.submodular import submodular_solve as docpx_submodular_solve
#from docpx.oabd import oabd_solve as docpx_oabd_solve 

# 
from cpx.conic import conic_solve as cpx_conic_solve
from cpx.gbdlin import gbdlin_solve as cpx_gbdlin_solve
from cpx.submodular import submodular_solve as cpx_submodular_solve
from cpx.submodularlifted import submodular_solve as cpx_submodular_solve_lifted
from cpx.oabd import oabd_solve as cpx_oabd_solve
from cpx.oabdspcpx2 import oabdspcpx_solve as cpx_oabdspcpx_solve
from cpx.oabdspcpxmultii import oabdspcpx_solve as cpx_oabdspcpxmultii_solve
import sys
import tracemalloc

#call line: python3.10 methods-gm.py 800 100 1.e3 1 2000 True

def main():
    
    nc = int(sys.argv[1])
    nf = int(sys.argv[2])
    sf = float(sys.argv[3])
    gm = int(sys.argv[4])
    fc = float(sys.argv[5])
    ty = bool(sys.argv[6])
    np = int(sys.argv[7])
     

    #data = CData(number_of_customers=100,number_of_facilities=100,u_scale_factor=1.e6,gamma=5,is_uniform_gamma=False)
    #data = CData(number_of_customers=75,number_of_facilities=50,u_scale_factor=1.e6,gamma=3,fixed_cost=1000,is_uniform_gamma=True)
    #data = CData(number_of_customers=100,number_of_facilities=70,u_scale_factor=1.e5,gamma=2,fixed_cost=1000,is_uniform_gamma=True)
    #data = CData(number_of_customers=75,number_of_facilities=50,u_scale_factor=1.e6,gamma=2,fixed_cost=500,is_uniform_gamma=False)
    #data = CData(number_of_customers=100,number_of_facilities=100,u_scale_factor=1.e4,gamma=5,fixed_cost=1000,is_uniform_gamma=False)
    #data = CData(number_of_customers=400,number_of_facilities=100,u_scale_factor=1.e6,gamma=5,fixed_cost=1000,is_uniform_gamma=False)
    data = CData(number_of_customers=nc,number_of_facilities=nf,u_scale_factor=sf,gamma=gm,fixed_cost=fc,is_uniform_gamma=ty,num_pre_cuts=np)

    # --------------------------------------
    # gurobi
    # --------------------------------------
    #print()
    #print("-"*100)
    #print(">>>>> gurobi <<<<<<")
    #print()
    # -------------------------
    # conic 
    # -------------------------
    #grb_conic_solve(data)
    #print()
    
    # -------------------------
    # gbdlin 
    # -------------------------
    #grb_gbdlin_solve(data)
    #print()

    # -------------------------
    # oabd 
    # -------------------------
    #grb_oabd_solve(data)
    #print()

    # -------------------------
    # submodular 
    # -------------------------
    #grb_submodular_solve(data)    
    #print()


    # --------------------------------------
    # cplex
    # --------------------------------------
    #print()
    #print("-"*100)
    #print(">>>>> cplex <<<<<<")
    #print()
    # -------------------------
    # conic 
    # -------------------------
    print()
    tracemalloc.start()
    cpx_conic_solve(data)
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory: {current / 10**6:.2f} MB; Peak memory: {peak / 10**6:.2f} MB")
    tracemalloc.stop()
    '''
    # --------------------------------------
    # oabd
    # --------------------------------------
    print()
    tracemalloc.start()
    cpx_oabd_solve(data)
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory: {current / 10**6:.2f} MB; Peak memory: {peak / 10**6:.2f} MB")
    tracemalloc.stop()
    # --------------------------------------
    # oabd
    # --------------------------------------
    print()
    print('-'* 100)
    tracemalloc.start()
    cpx_oabdspcpx_solve(data)
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory: {current / 10**6:.2f} MB; Peak memory: {peak / 10**6:.2f} MB")
    tracemalloc.stop()
    # --------------------------------------
    # oabd multi i sp cpx
    # --------------------------------------
    print()
    print('-'* 100)
    tracemalloc.start()
    cpx_oabdspcpxmultii_solve(data)
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory: {current / 10**6:.2f} MB; Peak memory: {peak / 10**6:.2f} MB")
    tracemalloc.stop()
    '''
    # --------------------------------------
    # sumodular
    # --------------------------------------
    #print()
    #cpx_submodular_solve(data)

    # --------------------------------------
    # sumodular-lifted
    # --------------------------------------
    #print()
    #cpx_submodular_solve_lifted(data)

    # --------------------------------------
    # gbdlin
    # --------------------------------------
    #print()
    #cpx_gbdlin_solve(data)

    # --------------------------------------
    # docplex
    # --------------------------------------
    #print()
    #print("-"*100)
    #print(">>>>> docplex <<<<<<")
    #print()
    # -------------------------
    # conic formulation 
    # -------------------------
    #print()
    #docpx_conic_solve(data)

    # -------------------------
    # oabd 
    # -------------------------
    #print()
    #docpx_oabd_solve(data)

    # -------------------------
    # submodular 
    # -------------------------
    #print()
    #docpx_submodular_solve(data)

    # -------------------------
    # gbdlin 
    # -------------------------
    #print()
    #docpx_gbdlin_solve(data)

if __name__ == '__main__':
    main()
