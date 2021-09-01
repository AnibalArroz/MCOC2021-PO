# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 11:44:04 2021

@author: Nicolas
"""




from numpy import zeros, float64
import numpy as np
import scipy.sparse as sparse

 
    
    
def matriz_laplaciana(N, t=float64):
    d = sparse.eye(N, N, 1,  dtype = t) - sparse.eye(N, N, 1, dtype = t)
    return 2*sparse.eye(N, dtype=t) -d - d.T




N =10


A = matriz_laplaciana(N)

#print (f"A = \n{A}:")


# Acsr = sparse.csr_matrix(A)
# Acsc = sparse.csc_matrix(A)
# Acoo = sparse.coo_matrix(A)
# Adia = sparse.dia_matrix(A)
# Alil = sparse.lil_matrix(A)

# print (Acsr)
# print (Acsc)
# print (Acoo)
# print (Adia)
# print (Alil)


#print (f)

