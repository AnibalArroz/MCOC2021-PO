
from numpy import half, single, double, longdouble, zeros, eye

def laplaciana(N, dtype):
    A = zeros((N, N), dtype=dtype)
    
    for i in range (N):
        A[i,i]=2
        for j in range (max(0, i)):
            if i == j:
                A[i,i]= 2
            elif abs(i-j) == 1:
                A[i,j]= -1
                A[j,i]= -1
    return A





















