# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 00:45:48 2021

@author: Nicolas
"""


import numpy as np
from time import perf_counter
from numpy import half, single, double, longdouble, zeros, eye
from numpy import float16, float32, float64
from numpy.linalg import inv
from laplaciana import laplaciana
import matplotlib.pylab as plt



N = 1000

Ns = [10, 15, 20, 30, 40, 50, 60, 70, 80, 100, 130, 160, 200, 300, 400, 500, 650, 800, 900, 1000, 1500, 2000, 2500, 3500, 5000, 6000, 8000, 10000]


t0 = perf_counter()
Ms = []
dt = []

f = open("scipy_numpy_half.txt", "a")



for N in Ns:
    
    # A = zeros((N, N))
    # Am1 = inv(A)
    
    t1 = perf_counter()
    A = laplaciana(N, dtype=half)
    t2 = perf_counter()
    
    #print (f"{A}")
    #exit(A)
    
    Aml = inv(A)
    t3 = perf_counter()
    
    #print (Aml)
    
    
    dt_ensamblaje = t2 - t1
    dt_inversion = t3 - t2
    
    dt.append(dt_inversion)
    
    bytes_total = (A.nbytes + Aml.nbytes)*(10**(-6))
    
    Ms.append(bytes_total)
    
    f.write(f"Matriz de {N}x{N}"+"\n")
    f.write(f"Uso memoria: {bytes_total} bytes MB"+"\n")
    f.write(f"Tiempo inversion: {dt_inversion} s"+"\n")
    
    
    print ("Matriz de ", N)
    print(f"Uso memoria: {bytes_total} bytes MB")   
    #print(f"Tiempo ensamblaje: {dt_ensamblaje} s")
    print(f"Tiempo inversion: {dt_inversion} s")
    
f.close()


plt.figure(1)
plt.subplot(2, 1, 1)
#plt.loglog(Ms, marker = "o")
plt.ylabel("Tiempo Transcurrido (s)")
plt.title("Rendimiento inv")
plt.loglog(Ms, dt,  marker = "o")
plt.grid(True)

plt.show()

t4 = perf_counter()

dt_total= t4 - t0
print(f"Tiempo ejecucion: {dt_total} s")