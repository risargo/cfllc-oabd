import sys

ZERO = 1e-6
GUROBI_STATUS_MSG ={\
   1 :  "LOADED",   
   2 :  "OPTIMAL",    
   3 :  "INFEASIBLE",    
   4 :  "INF_OR_UNBD",    
   5 :  "UNBOUNDED",    
   6 :  "CUTOFF",    
   7 :  "ITERATION_LIMIT",
   8 :  "NODE_LIMIT",    
   9 :  "TIME_LIMIT",    
  10 :  "SOLUTION_LIMIT",    
  11 :  "INTERRUPTED",    
  12 :  "NUMERIC",    
  13 :  "SUBOPTIMAL",    
  14 :  "INPROGRESS",    
  15 :  "USER_OBJ_LIMIT",    
  16 :  "WORK_LIMIT",    
  17 :  "MEM_LIMIT"}

def print_error_msg(error_msg):
    print(f"error message: {error_msg}")
    sys.exit()
