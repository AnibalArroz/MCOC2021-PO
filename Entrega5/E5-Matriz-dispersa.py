# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 22:05:30 2021

@author: Nicolas
"""




import numpy as np
from time import perf_counter
from numpy import half, single, double, longdouble, zeros, eye
from numpy import float16, float32, float64
from scipy.linalg import inv
from matriz_laplaciana import matriz_laplaciana
import matplotlib.pylab as plt
import scipy.sparse as sparse
from psutil import virtual_memory

N = 10

#Ns = [10, 15, 20, 30, 40, 50, 60, 70, 80, 100, 130, 160, 200, 300, 400, 500, 650, 800, 900, 1000, 1500, 2000, 2500, 3500, 5000, 6000, 8000, 10000, 20000, 40000]
Ns = [10,20,  50, 100, 200,300, 500,800, 1000, 2000, 5000, 10000, 12000, 15000, 20000, 40000, 50000, 100000, 200000, 500000, 1000000, 1500000, 2000000, 3000000, 4000000, 10000000, 20000000 ]
#Ns = [10, 20, 50, 100,200,  500, 1000, 2000]

Ms = []
dt = []

f = open("Matriz_dispersa_E5.txt", "a")

for N in Ns:
    
    t1 = perf_counter()
    A = matriz_laplaciana(N)    
    B = matriz_laplaciana(N)
    Acsr = sparse.csr_matrix(A)
    Bcsr = sparse.csr_matrix(B)   
    t2 = perf_counter()
    
    
    t3 = perf_counter()
    x = Acsr@Bcsr
    #x = np.matmul(Acsr,Bcsr)
    t4 = perf_counter()
    
    #print (x)
    
    dt_ensamblaje = t2 - t1
    Ms_solucion = t4 - t3
    
    dt.append(dt_ensamblaje)
    Ms.append(Ms_solucion)
    
    f.write(f"Matriz de {N}x{N}"+"\n")
    f.write(f"Tiempo ensamblaje: {dt_ensamblaje} s"+"\n")
    f.write(f"Tiempo solución: {Ms_solucion} s"+"\n")
    
    # print ("Matriz de ", N)
    # print(f"Tiempo ensamblaje: {dt_ensamblaje} s")
    # print(f"Tiempo solución: {dt_solucion} s")
    
f.close()



# plt.figure(1)
g1 = plt.subplot(2, 1, 1)
plt.ylabel("Tiempo de Ensamblado")
plt.loglog(Ns, dt,  "k", marker = "o")

#Yticks (Tiempo Transcurrido (s))
labelsY1 = ["0,1 ms", "1 ms", "10 ms", "0,1 s", "1 s", "10 s", "1 min", "10 min"]
labels_Y1 = [10**(-4), 10**(-3), 10**(-2), 10**(-1), 10**(0), 10**(1), 10**(2), 10**(3) ]
plt.yticks(labels_Y1, labelsY1)

#Xticks
labelsX = ["", "", "", "", "", "", "", "", "", "", "" ]
xlabel2 = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
plt.xticks(xlabel2, labelsX, rotation = 45)




#   LINEAS INTERMITENTES

#LINEA BLANCA 
plt.plot([10, 20000000],[10**(3),10**(3)], linestyle='--', color='W', label="Constante" )

#LINEA AZUL (CONSTANTE)
plt.plot([10, 20000000],[4.105, 4.105], linestyle='--', color='royalblue', label="Constante" )

# LINEA NARANJA (ON)
plt.plot([10, 20000000],[0.001533, 4.105], linestyle='--', color='gold', label="O(N)")

# LINEA VERDE (ON2)
gE1 = (20000000)*(((0.00000001/4.105)**(1/2)))
plt.plot([gE1, 20000000],[0.00001,4.105], linestyle='--', color='g', label="O(N2)")

# LINEA ROJA (ON3)
rE1 = (20000000)*(((0.00000001/4.105)**(1/3)))
plt.plot([rE1, 20000000],[0.00001,4.105], linestyle='--', color='r', label="O(N3)")

# LINEA MORADA (ON4)
mE1 = (20000000)*(((0.00000001/4.105)**(1/4)))
plt.plot([mE1, 20000000],[0.00001,4.105], linestyle='--', color='m', label="O(N4)")





# plt.figure(2)
plt.subplot(2, 1, 2)
plt.ylabel("Tiempo de Solución ")
plt.xlabel("Tamaño Matriz N")
plt.loglog(Ns, Ms,  "k",  marker = "o")

#Yticks (Tiempo Transcurrido (s))
labelsY1 = ["0,1 ms", "1 ms", "10 ms", "0,1 s", "1 s", "10 s", "1 min", "10 min"]
labels_Y1 = [10**(-4), 10**(-3), 10**(-2), 10**(-1), 10**(0), 10**(1), 10**(2), 10**(3) ]
plt.yticks(labels_Y1, labelsY1)

#Xticks
labelsX = ["10", "20", "50", "100", "200", "500", "1000", "2000", "5000", "10000", "20000" , "50000", "100000", "200000",  "500000", "1000000","2000000",  "5000000", "10000000", "20000000"]
xlabel2 = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000, 2000000, 5000000, 10000000, 20000000]
plt.xticks(xlabel2, labelsX, rotation = 45)




#   LINEAS INTERMITENTES

#LINEA BLANCA 
plt.plot([10, 20000000],[10**(3),10**(3)], linestyle='--', color='W', label="Constante" )

#LINEA AZUL (CONSTANTE)
plt.plot([10, 20000000],[1.8301,1.8301], linestyle='--', color='royalblue', label="Constante" )

# LINEA NARANJA (ON)
plt.plot([10, 20000000],[0.0001623,1.8301], linestyle='--', color='gold', label="O(N)")

# LINEA VERDE (ON2)
gE = (20000000)*(((0.00000001/1.8301)**(1/2)))
plt.plot([gE, 20000000],[0.00001,1.8301], linestyle='--', color='g', label="O(N2)")

# LINEA ROJA (ON3)
rE = (20000000)*(((0.00000001/1.8301)**(1/3)))
plt.plot([rE, 20000000],[0.00001,1.8301], linestyle='--', color='r', label="O(N3)")

# LINEA MORADA (ON4)
mE = (20000000)*(((0.00000001/1.8301)**(1/4)))
plt.plot([mE, 20000000],[0.00001,1.8301], linestyle='--', color='m', label="O(N4)")



plt.legend(('', '','Constante', 'O(N)', 'O(N2)', 'O(N3)', 'O(N4)'),
           loc='lower left')


plt.savefig("grafico_Matriz_dispersa.png")


plt.show()
