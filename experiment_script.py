#Felipe Gonzalez Casabianca
#Source code for the experiments of the first project of the course: Introduccion to Convex Optimization 
#This script uses exclusevly the library fancyimpute, focusing on the NuclearNormMinimization, since it implements the approach explained in the project

#For simple syntax divition
from __future__ import division
#The main library
from fancyimpute import NuclearNormMinimization
#For the definition of matrices:
import numpy as np
#For pseudorandom number generation 
import random as rand
#For math operation
import math

total = 10


m = 10*np.asmatrix( np.random.rand(total,total))
m_incomplete = np.copy(m)

for i in range(0,10):
    m_incomplete[rand.randint(0,total-1), rand.randint(0,total-1)] = np.nan

m_complete = NuclearNormMinimization().complete(m_incomplete)    
    
print(m)
print(m_complete)
