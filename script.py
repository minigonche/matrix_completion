# Felipe Gonzalez Casabianca
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
#For math simple operations
import math
#For Graphing
import plotly.plotly as py
import plotly.graph_objs as go
py.sign_in('minigonche', '8cjqqmkb4o') #This api-key has been changed already



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

    

#Gets a matrix wth a certain number of observation of the given matrix.
# this values are chosen uniformly at random and the rest of the values are
# marked as NaN
def get_observed_matrix(M, num_observations):
    
    n = M.shape[0]
    total = n*n
    empty_matrix = np.empty((n,n,))
    empty_matrix[:] = np.NAN
    empty_matrix = np.matrix(empty_matrix)

    m_origin = np.copy(M)
    m_dest = np.copy(empty_matrix)
    flag = False
    
    # If the total amount of observations is to large, is better to simply
    # remove values instead of adding
    if(total/2 < num_observations):
        m_origin = np.copy(empty_matrix)
        m_dest = np.copy(M) 
        num_observations = total - num_observations
        flag = True
        
    for i in range(num_observations):
        coor = [rand.randint(0,n-1),rand.randint(0,n-1)]
        # Selects randomly untils the coordinate selected is idle
        while(np.isnan(m_dest[coor[0],coor[1]]) == flag):
            coor = [rand.randint(0,n-1),rand.randint(0,n-1)]
            
        #Updates the coordinate
        m_dest[coor[0],coor[1]] = m_origin[coor[0],coor[1]]
    
    return m_dest        


# ----------------------------------------------------------------------
# ----------------------  Experiments  ---------------------------------
# ----------------------------------------------------------------------

# Graphs the mean percentage of the recovered matrix vs the amount of 
# observed entries

def graph_percentage_vs_observed(dim, ite, tries, r, tol, fast_but_approximate):

    """
        Parameters
        ----------
        dim : int
            The dimention of the matrix (square)
        ite : int 
            The number of iterations per observations
        tries : int
            The number of values (evenly distributed) that the number of 
            observations will take
        r : int
            The number of singular vectors
        tol : float
            The tolerance for two values to be considered equal
        fast_but_approximate : bool
            Use the faster but less accurate Splitting Cone Solver (part of
            FancyImpute)    
    """
    
    if(dim <= 1):
        raise ValueError('''The dimention must be larger than 1''')
        
    total = dim*dim        
    skip = (dim*dim - 1)/tries
    m_values = np.arange(1,total + skip, skip)
    
    #The percentages
    per = []
    
    for m in m_values:
        for i in range(ite):
            #Complete Random Matrix
            initial_matrix = random_matrix(dim, r)
            #Incomplete Matrix
            incomplete_matrix = get_observed_matrix(initial_matrix, m)
            #Guessed Matrix
            guessed_matrix = NuclearNormMinimization(fast_but_approximate=
            fast_but_approximate).complete(incomplete_matrix)
            
            #The boolean matrix indicating if two coordenates are equal 
            # (with the given tolerance)
            bool_matrix = np.isclose(initial_matrix, guessed_matrix,atol = tol)
            percentage = sum(sum(bool_matrix))/total
            #Saves percentage
            per.append(percentage)
    
    trace_m = go.Scatter(x = m_values, y =  m_values/total, mode = 'Observed Percentage')
    trace_r = go.Scatter(x = m_values, y =  per, mode = 'Recovered Percentage')
    #plot_url = py.iplot([trace_m,trace_r])  


graph_percentage_vs_observed(dim = 6, ite = 2, tries = 5, r= 5, tol = 0.001, fast_but_approximate = True)            
            
            

