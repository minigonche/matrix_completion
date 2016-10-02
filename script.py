#Felipe Gonzalez Casabianca
# Source code for the experiments of the first project of the course: 
# Introduccion to Convex Optimization 
# This script uses exclusevly the library fancyimpute, focusing on the 
# NuclearNormMinimization, since it implements the approach explained in 
# the project

#-------- Script Imports ------------
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


#The gram schmit coeficient
def gs_cofficient(v1, v2):
    return np.dot(v2, v1) / np.dot(v1, v1)

#Multiply a vector by a coefficient
def multiply(cofficient, v):
    return map((lambda x : x * cofficient), v)

#Calculates the projection of v2 onto v1
def proj(v1, v2):
    return multiply(gs_cofficient(v1, v2) , v1)
    
#Calclates the norm of the given vector    
def norm(v1):
    return math.sqrt(np.dot(v1, v1))
    
#Generates a random vector    
def random_vector(dim):
    vec = []
    for i in range(dim):
        vec.append(rand.uniform(-1, 1))
        
    if(norm(vec) == 0):
        return random_vector(dim)
        
    return vec

#Generates a random positive vector
def random_postive_vector(dim,max_value):
    vec = []
    for i in range(dim):
        vec.append(rand.uniform(0, max_value))
        
    if(norm(vec) == 0):
        return random_vector(dim)
        
    return vec

#Genrates a random orthonormal set of vectors
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
        raise ValueError('''The size of the linearly independent set cannot be 
        grater than the dimention of the vectors''')

    Y = []
    for i in range(n):
        temp_vec = random_vector(dim)
        for inY in Y :
            # Finds the projection
            proj_vec = proj(inY, temp_vec)
            # Substracts the vector
            temp_vec = map(lambda x, y : x - y, temp_vec, proj_vec)
        #Normalizes the vector    
        temp_vec = multiply(1/norm(temp_vec), temp_vec)    
        Y.append(temp_vec)
    return Y

#-------- Random Orthogonal Model Matrix ------------
# This code construct a random matrix using the random
# orthgonal model

#Genrates a random matrix using the random orhonormal model
def random_matrix(dim, r):
    """
        Parameters
        ----------
        dim : int
            The dimention of the matrix (square)
        r : int
            The number of singular vectors
    """
    #First set of orthonormal vectors (single vectors)
    set1 = random_ortho_set(r,dim)
    #Second set of orthonormal vectors (single vectors)
    set2 = random_ortho_set(r,dim)
    #Set of single values
    sing_values = random_postive_vector(r,50)
    
    #generates the matrix
    M = sing_values[0]*np.matrix(set1[0]).transpose()*np.matrix(set2[0])
    for i in range(1,r):
        temp = np.matrix(set1[i]).transpose()*np.matrix(set2[i])
        M = M + sing_values[i]*temp
    
    return M

print(random_matrix(5,3))    

