# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 10:45:10 2021

@author: Nicolas
"""


import numpy as np
import scipy.sparse as sparse


def matriz_laplaciana (N, t = np.float32):
    return sparse.eye(N, dtype=t) - sparse.eye(N, N, 1, dtype=t)
  # e=np.eye(N)-np.eye(N, N, 1)
  #   return t(e+e.T)
    
    
    

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





