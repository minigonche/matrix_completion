#Felipe Gonzalez Casabianca
#Source code for the experiments of the first project of the course: Introduccion to Convex Optimization 

#This Script escecutes the different experiments surrounding matrix completion problem for random matrices and adjacency matrices of graphs

#This script uses exclusevly the library fancyimpute, focusing on the NuclearNormMinimization, since it implements the approach explained in the project


from fancyimpute1 import NuclearNormMinimization
#For the definition of matrices:
import numpy as np
#For pseudorandom number generation 
import random as rand

total = 10


m = 10*np.asmatrix( np.random.rand(total,total))
m_incomplete = np.copy(m)

for i in range(0,5):
    m_incomplete[rand.randint(0,total-1), rand.randint(0,total-1)] = np.nan

m_complete = NuclearNormMinimization(fast_but_approximate=False).complete(m_incomplete)    
    
print(m)
print(m_complete)
