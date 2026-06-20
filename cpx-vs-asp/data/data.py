import numpy as np
from scipy.spatial.distance import cdist
'''
   instance data
   
   demand
   (x,y) customer coordinates : location_customer_xy 
   (x,y) facility coordinates : facility_location_xy 
   distance 
   fixed costs : facility_cost 
 
   facility utilities : u
   competitor's utility : u0
   
   pi : u/u0 
   
   choice rule size : gamma
'''

class CData():
   def __init__(self,number_of_customers, number_of_facilities, u_scale_factor=1e5, gamma=5, fixed_cost = 2000,is_uniform_gamma=True,num_pre_cuts=25):
       np.random.seed(1)

       demand = np.random.randint(10, 1000,number_of_customers)
       location_customer_xy = np.random.uniform(0,1000,(number_of_customers,2))
       facility_location_xy = np.random.uniform(0,1000,(number_of_facilities,2))
       distance = cdist(location_customer_xy,facility_location_xy)
       #facility_cost = np.full((number_of_facilities,),2000)
       facility_cost = np.full((number_of_facilities,),fixed_cost).astype(float)

       u = np.rint(u_scale_factor/distance**2).astype(int)
       u0 = max(1, np.rint(u_scale_factor/50**2).astype(int) )  

       #u = 1/distance**2
       #u0 = 1/50**2
       
       nzeros = np.count_nonzero(u)
       print(f"u non zero elements: {nzeros} / {u.size} ({100.0 * nzeros/u.size:6.2f} %) ")
       pi = u/u0

       self.ni = number_of_customers
       self.nj = number_of_facilities
       self.u_scale_factor = u_scale_factor
       self.npc = num_pre_cuts
       I,J = range(self.ni),range(self.nj)

       if is_uniform_gamma == True:
          self.gamma = np.full((number_of_customers,),gamma)
       else:
          self.gamma = np.random.randint(1,gamma+1,number_of_customers)

       self.cxy = location_customer_xy
       self.fxy = facility_location_xy
       self.b = demand.astype(float)
       self.total_b = demand.sum()
       self.f = facility_cost
       self.pi = pi
       self.u = u
       self.u0 = u0
       self.sorted_idpi = np.fliplr(np.argsort(pi,axis=1))
       self.sorted_pi = np.sort(pi,axis=1)[:,::-1] 

       self.sorted_idu = np.fliplr(np.argsort(u,axis=1))
       self.sorted_u = np.sort(u,axis=1)[:,::-1] 

       self.sorted_inu = np.argsort(u,axis=1)       
       dist = cdist(self.pi.T,self.pi.T)
       self.neighbor = np.argsort(dist,axis=1)
