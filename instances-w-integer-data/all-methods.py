from data.data import CData 
from cpx.conic import conic_solve as cpx_conic_solve
from cpx.gbdlin import gbdlin_solve as cpx_gbdlin_solve
from cpx.submodular import submodular_solve as cpx_submodular_solve
from cpx.submodularlifted import submodular_solve as cpx_submodular_solve_lifted
from cpx.oabd import oabd_solve as cpx_oabd_solve
import sys

#call line: python3.10 methods-gm.py 800 100 1.e3 1 2000 True

def main():

    nc = int(sys.argv[1])
    nf = int(sys.argv[2])
    sf = float(sys.argv[3])
    gm = int(sys.argv[4])
    fc = float(sys.argv[5])
    ty = bool(sys.argv[6])
    np = int(sys.argv[7])

    data = CData(number_of_customers=nc,number_of_facilities=nf,u_scale_factor=sf,gamma=gm,fixed_cost=fc,is_uniform_gamma=ty,num_pre_cuts=np)

    # -------------------------
    # conic 
    # -------------------------
    cpx_conic_solve(data)
    print()

    # --------------------------------------
    # oabd
    # --------------------------------------
    print()
    cpx_oabd_solve(data)

    # --------------------------------------
    # sumodular
    # --------------------------------------
    print()
    cpx_submodular_solve(data)

    # --------------------------------------
    # sumodular-lifted
    # --------------------------------------
    print()
    cpx_submodular_solve_lifted(data)

    # --------------------------------------
    # gbdlin
    # --------------------------------------
    print()
    cpx_gbdlin_solve(data)


if __name__ == '__main__':
    main()
