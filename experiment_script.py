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




#-------- Random Orthogonal Vector Sets ------------
# This code constructs pseudorandom sets of orthogonal vectors,
# using the gram schmit process to orthogonalize the random vectors



def gs_cofficient(v1, v2):
    return np.dot(v2, v1) / np.dot(v1, v1)

def multiply(cofficient, v):
    return map((lambda x : x * cofficient), v)

def proj(v1, v2):
    return multiply(gs_cofficient(v1, v2) , v1)
    
def norm(v1):
    return math.sqrt(np.dot(v1, v1))
    
def random_vector(dim):
    vec = []
    for i in range(dim):
        vec.append(rand.uniform(-1, 1))
        
    if(norm(vec) == 0):
        return random_vector(dim)
        
    return vec

def random_psotive_vector(dim,max_value):
    vec = []
    for i in range(dim):
        vec.append(rand.uniform(0, max_value))
        
    if(norm(vec) == 0):
        return random_vector(dim)
        
    return vec

def random_ortho_set(n, dim):
    """
        Parameters
        ----------
        n : int
            The number of vectors in the set
        dim : int
            The dimention of the vectors
    """
    if(n > dim):
        raise ValueError('The size of the linearly independent set cannot be grater than the dimention of the vectors')

    Y = []
    for i in range(n):
        temp_vec = random_vector(dim)
        for inY in Y :
            proj_vec = proj(inY, temp_vec)
            temp_vec = map(lambda x, y : x - y, temp_vec, proj_vec)
        temp_vec = multiply(1/norm(temp_vec), temp_vec)    
        Y.append(temp_vec)
    return Y


print np.array(random_ortho_set(2, 3))



total = 10


m = 10*np.asmatrix( np.random.rand(total,total))
m_incomplete = np.copy(m)

for i in range(0,10):
    m_incomplete[rand.randint(0,total-1), rand.randint(0,total-1)] = np.nan

m_complete = NuclearNormMinimization(fast_but_approximate=False).complete(m_incomplete)    
    
print(m)
print(m_complete)
