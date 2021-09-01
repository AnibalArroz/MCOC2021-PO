

CODIGOS

------------------------------------------------------------------------------------------
# LAPLACIANA


from numpy import zeros, float64
import numpy as np
import scipy.sparse as sparse

 
    
    
def matriz_laplaciana(N, t=float64):
    d = sparse.eye(N, N, 1,  dtype = t) - sparse.eye(N, N, 1, dtype = t)
    return 2*sparse.eye(N, dtype=t) -d - d.T




N =10


A = matriz_laplaciana(N)



----------------------------------------------------------------------------------------------------

-
-
-
-
-
-
-
-
-
-
-
-
-


# MATRIZ LLENA SOLVE

from numpy import double, ones
import scipy.sparse as sp
import scipy.sparse as sparse
import scipy.sparse.linalg as lin
from laplaciana import matriz_laplaciana
from time import perf_counter
import matplotlib.pylab as plt
import numpy as np
import matplotlib.pyplot as plt

Ns = [10, 20, 30, 50, 70, 100, 150, 200, 400, 600, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 7000, 10000, 15000, 30000, 50000, 100000, 200000, 500000, 700000, 1000000 ]
#Ns = [20, 50, 100, 200, 500, 800, 1000, 1500, 2000, 3000, 4000, 5000, 8000]


Ms = []
dt = []

f = open("Matriz__llena_solve_E6.txt", "a")


for N in Ns:
    
    
    
    t1 = perf_counter()
    
    A = matriz_laplaciana(N, double)
    B = ones(N, dtype=double)
    
    t2 = perf_counter()
    
    x = lin.spsolve(A,B)
    
    
    t3 = perf_counter()
    
    dt_ensamblaje = t2 - t1
    Ms_solucion = t3 - t2
    
    dt.append(dt_ensamblaje)
    Ms.append(Ms_solucion)
    
    f.write(f"Matriz de {N}x{N}"+"\n")
    f.write(f"Tiempo ensamblaje: {dt_ensamblaje} s"+"\n")
    f.write(f"Tiempo solución: {Ms_solucion} s"+"\n")
    
    # print ("Matriz de ", N)
    # print(f"Tiempo ensamblaje: {dt_ensamblaje} s")
    # print(f"Tiempo solución: {dt_solucion} s")
    
    # print (f"Tiempo de ensamblaje: {t2-t1}")
    # print (f"Tiempo de solucion: {t3-t2}")
    
f.close()
    
    


# plt.figure(1)
plt.subplot(2, 1, 1)
plt.ylabel("Tiempo de Ensamblado")
plt.loglog(Ns, dt, "k",  marker = "o")

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
plt.plot([10, 1000000],[10**(3),10**(3)], linestyle='--', color='W', label="Constante" )

#LINEA AZUL (CONSTANTE)
plt.plot([10, 1000000],[0.19, 0.19], linestyle='--', color='royalblue', label="Constante" )

# LINEA NARANJA (ON)
plt.plot([10, 1000000],[0.001644, 0.19], linestyle='--', color='gold', label="O(N)")

# LINEA VERDE (ON2)
gE1 = (1000000)*(((0.00000001/0.19)**(1/2)))
plt.plot([gE1, 1000000],[0.00001,0.19], linestyle='--', color='g', label="O(N2)")

# LINEA ROJA (ON3)
rE1 = (1000000)*(((0.00000001/0.19)**(1/3)))
plt.plot([rE1, 1000000],[0.00001,0.19], linestyle='--', color='r', label="O(N3)")

# LINEA MORADA (ON4)
mE1 = (1000000)*(((0.00000001/0.19)**(1/4)))
plt.plot([mE1, 1000000],[0.00001,0.19], linestyle='--', color='m', label="O(N4)")




# plt.figure(2)
plt.subplot(2, 1, 2)
plt.ylabel("Tiempo de Solución ")
plt.xlabel("Tamaño Matriz N")
plt.loglog(Ns, Ms, "k",  marker = "o")

#Yticks (Tiempo Transcurrido (s))
labelsY1 = ["0,1 ms", "1 ms", "10 ms", "0,1 s", "1 s", "10 s", "1 min", "10 min"]
labels_Y1 = [10**(-4), 10**(-3), 10**(-2), 10**(-1), 10**(0), 10**(1), 10**(2), 10**(3) ]
plt.yticks(labels_Y1, labelsY1)

#Xticks
labelsX = ["10", "20", "50", "100", "200", "500", "1000", "2000", "5000", "10000", "20000", "50000", "100000", "200000",  "500000", "1000000" ]
xlabel2 = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000]
plt.xticks(xlabel2, labelsX, rotation = 45)


#   LINEAS INTERMITENTES

#LINEA BLANCA 
plt.plot([10, 1000000],[10**(3),10**(3)], linestyle='--', color='W', label="Constante" )

#LINEA AZUL (CONSTANTE)
plt.plot([10, 1000000],[0.5187,0.5187], linestyle='--', color='royalblue', label="Constante" )

# LINEA NARANJA (ON)
plt.plot([10, 1000000],[10**(-4),0.5187], linestyle='--', color='gold', label="O(N)")

# LINEA VERDE (ON2)
gE = (1000000)*(((0.00000001/0.5187)**(1/2)))
plt.plot([gE, 1000000],[0.00001,0.5187], linestyle='--', color='g', label="O(N2)")

# LINEA ROJA (ON3)
rE = (1000000)*(((0.00000001/0.5187)**(1/3)))
plt.plot([rE, 1000000],[0.00001,0.5187], linestyle='--', color='r', label="O(N3)")

# LINEA MORADA (ON4)
mE = (1000000)*(((0.00000001/0.5187)**(1/4)))
plt.plot([mE, 1000000],[0.00001,0.5187], linestyle='--', color='m', label="O(N4)")


plt.legend(('', '','Constante', 'O(N)', 'O(N2)', 'O(N3)', 'O(N4)'),
           loc='lower left')


plt.savefig("grafico_Matriz_llena_solve_E6.png")

plt.show()

------------------------------------------------------------------------------------------------------------
-
-
-
-
-
-
-
-
-
-
-
-


# MATRIZ DISPERSA SOLVE

from numpy import double, ones
import scipy.sparse as sp
import scipy.sparse as sparse
import scipy.sparse.linalg as lin
from laplaciana import matriz_laplaciana
from time import perf_counter
import matplotlib.pylab as plt

Ns = [10, 20, 30, 50, 70, 100, 150, 200, 400, 600, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 7000, 10000, 15000, 30000, 50000, 100000, 200000, 500000, 700000, 1000000 ]



Ms = []
dt = []

f = open("Matriz_dispersa_solve_E6.txt", "a")


for N in Ns:
    
    
    
    t1 = perf_counter()
    
    A = matriz_laplaciana(N, double)
    Acsr = sp.csr_matrix(A)
    B = ones(N, dtype=double)
    
    t2 = perf_counter()
    
    x = lin.spsolve(Acsr,B)
    
    
    t3 = perf_counter()
    
    dt_ensamblaje = t2 - t1
    Ms_solucion = t3 - t2
    
    dt.append(dt_ensamblaje)
    Ms.append(Ms_solucion)
    
    f.write(f"Matriz de {N}x{N}"+"\n")
    f.write(f"Tiempo ensamblaje: {dt_ensamblaje} s"+"\n")
    f.write(f"Tiempo solución: {Ms_solucion} s"+"\n")
    
    # print ("Matriz de ", N)
    # print(f"Tiempo ensamblaje: {dt_ensamblaje} s")
    # print(f"Tiempo solución: {dt_solucion} s")
    
    # print (f"Tiempo de ensamblaje: {t2-t1}")
    # print (f"Tiempo de solucion: {t3-t2}")
    
f.close()
    
    


# plt.figure(1)
plt.subplot(2, 1, 1)
plt.ylabel("Tiempo de Ensamblado")
plt.loglog(Ns, dt, "k", marker = "o")

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
plt.plot([10, 1000000],[10**(3),10**(3)], linestyle='--', color='W', label="Constante" )

#LINEA AZUL (CONSTANTE)
plt.plot([10, 1000000],[0.1878, 0.1878], linestyle='--', color='royalblue', label="Constante" )

# LINEA NARANJA (ON)
plt.plot([10, 1000000],[0.00371, 0.1878], linestyle='--', color='gold', label="O(N)")

# LINEA VERDE (ON2)
gE1 = (1000000)*(((0.00000001/0.1878)**(1/2)))
plt.plot([gE1, 1000000],[0.00001,0.1878], linestyle='--', color='g', label="O(N2)")

# LINEA ROJA (ON3)
rE1 = (1000000)*(((0.00000001/0.1878)**(1/3)))
plt.plot([rE1, 1000000],[0.00001,0.1878], linestyle='--', color='r', label="O(N3)")

# LINEA MORADA (ON4)
mE1 = (1000000)*(((0.00000001/0.1878)**(1/4)))
plt.plot([mE1, 1000000],[0.00001,0.1878], linestyle='--', color='m', label="O(N4)")



# plt.figure(2)
plt.subplot(2, 1, 2)
plt.ylabel("Tiempo de Solución ")
plt.xlabel("Tamaño Matriz N")
plt.loglog(Ns, Ms, "k", marker = "o")

#Yticks (Tiempo Transcurrido (s))
labelsY1 = ["0,1 ms", "1 ms", "10 ms", "0,1 s", "1 s", "10 s", "1 min", "10 min"]
labels_Y1 = [10**(-4), 10**(-3), 10**(-2), 10**(-1), 10**(0), 10**(1), 10**(2), 10**(3) ]
plt.yticks(labels_Y1, labelsY1)

#Xticks
labelsX = ["10", "20", "50", "100", "200", "500", "1000", "2000", "5000", "10000", "20000" ]
xlabel2 = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
plt.xticks(xlabel2, labelsX, rotation = 45)



#   LINEAS INTERMITENTES

#LINEA BLANCA 
plt.plot([10, 1000000],[10**(3),10**(3)], linestyle='--', color='W', label="Constante" )

#LINEA AZUL (CONSTANTE)
plt.plot([10, 1000000],[0.5032,0.5032], linestyle='--', color='royalblue', label="Constante" )

# LINEA NARANJA (ON)
plt.plot([10, 1000000],[0.00019,0.5032], linestyle='--', color='gold', label="O(N)")

# LINEA VERDE (ON2)
gE = (1000000)*(((0.00000001/0.5032)**(1/2)))
plt.plot([gE, 1000000],[0.00001,0.5032], linestyle='--', color='g', label="O(N2)")

# LINEA ROJA (ON3)
rE = (1000000)*(((0.00000001/0.5032)**(1/3)))
plt.plot([rE, 1000000],[0.00001,0.5032], linestyle='--', color='r', label="O(N3)")

# LINEA MORADA (ON4)
mE = (1000000)*(((0.00000001/0.5032)**(1/4)))
plt.plot([mE, 1000000],[0.00001,0.5032], linestyle='--', color='m', label="O(N4)")


plt.legend(('', '','Constante', 'O(N)', 'O(N2)', 'O(N3)', 'O(N4)'),
           loc='lower left')


plt.savefig("grafico_Matriz_dispersa_solve_E6.png")


plt.show()

----------------------------------------------------------------------------------------------------
-
-
-
-
-
-
-
-
-
-
-
-
-



# MATRIZ INVERTIDA LLENA

from numpy import double, ones, array
import scipy.sparse as sp
import scipy.sparse as sparse
import scipy.sparse.linalg as lin
from laplaciana import matriz_laplaciana
from time import perf_counter
import matplotlib.pylab as plt

Ns = [10, 20, 30, 50, 70, 100, 150, 200, 400, 600, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 7000, 10000, 15000, 20000]



Ms = []
dt = []

f = open("Matriz_llena_inv_E6.txt", "a")


for N in Ns:
    
    
    
    t1 = perf_counter()
    
    A = matriz_laplaciana(N, double)
   
    
    t2 = perf_counter()
    
    Ainv = lin.inv(A)
    #print (array(Ainv))
    
    t3 = perf_counter()

    dt_ensamblaje = t2 - t1
    Ms_solucion = t3 - t2
    
    dt.append(dt_ensamblaje)
    Ms.append(Ms_solucion)
    
    f.write(f"Matriz de {N}x{N}"+"\n")
    f.write(f"Tiempo ensamblaje: {dt_ensamblaje} s"+"\n")
    f.write(f"Tiempo solución: {Ms_solucion} s"+"\n")
    
    # print ("Matriz de ", N)
    # print(f"Tiempo ensamblaje: {dt_ensamblaje} s")
    # print(f"Tiempo solución: {dt_solucion} s")
    
    # print (f"Tiempo de ensamblaje: {t2-t1}")
    # print (f"Tiempo de solucion: {t3-t2}")
    
f.close()
    
    


# plt.figure(1)
plt.subplot(2, 1, 1)
plt.ylabel("Tiempo de Ensamblado")
plt.loglog(Ns, dt, "k",marker = "o")

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
plt.plot([10, 20000],[10**(3),10**(3)], linestyle='--', color='W', label="Constante" )

#LINEA AZUL (CONSTANTE)
plt.plot([10, 20000],[0.003438, 0.003438], linestyle='--', color='royalblue', label="Constante" )

# LINEA NARANJA (ON)
plt.plot([10, 20000],[0.0019, 0.003438], linestyle='--', color='gold', label="O(N)")

# LINEA VERDE (ON2)
gE1 = (20000)*(((0.00000001/0.003438)**(1/2)))
plt.plot([gE1, 20000],[0.00001,0.003438], linestyle='--', color='g', label="O(N2)")

# LINEA ROJA (ON3)
rE1 = (20000)*(((0.00000001/0.003438)**(1/3)))
plt.plot([rE1, 20000],[0.00001,0.003438], linestyle='--', color='r', label="O(N3)")

# LINEA MORADA (ON4)
mE1 = (20000)*(((0.00000001/0.003438)**(1/4)))
plt.plot([mE1, 20000],[0.00001,0.003438], linestyle='--', color='m', label="O(N4)")






# plt.figure(2)
plt.subplot(2, 1, 2)
plt.ylabel("Tiempo de Solución ")
plt.xlabel("Tamaño Matriz N")
plt.loglog(Ns, Ms, "k", marker = "o")

#Yticks (Tiempo Transcurrido (s))
labelsY1 = ["0,1 ms", "1 ms", "10 ms", "0,1 s", "1 s", "10 s", "1 min", "10 min"]
labels_Y1 = [10**(-4), 10**(-3), 10**(-2), 10**(-1), 10**(0), 10**(1), 10**(2), 10**(3) ]
plt.yticks(labels_Y1, labelsY1)

#Xticks
labelsX = ["10", "20", "50", "100", "200", "500", "1000", "2000", "5000", "10000", "20000" ]
xlabel2 = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
plt.xticks(xlabel2, labelsX, rotation = 45)


#   LINEAS INTERMITENTES

#LINEA BLANCA 
plt.plot([10, 20000],[10**(3),10**(3)], linestyle='--', color='W', label="Constante" )

#LINEA AZUL (CONSTANTE)
plt.plot([10, 20000],[17.5124867,17.5124867], linestyle='--', color='royalblue', label="Constante" )

# LINEA NARANJA (ON)
plt.plot([10, 20000],[0.004744,17.5124867], linestyle='--', color='gold', label="O(N)")

# LINEA VERDE (ON2)
gE = (20000)*(((0.00000001/17.5124867)**(1/2)))
plt.plot([gE, 20000],[0.00001,17.5124867], linestyle='--', color='g', label="O(N2)")

# LINEA ROJA (ON3)
rE = (20000)*(((0.00000001/17.5124867)**(1/3)))
plt.plot([rE, 20000],[0.00001,17.5124867], linestyle='--', color='r', label="O(N3)")

# LINEA MORADA (ON4)
mE = (20000)*(((0.00000001/17.5124867)**(1/4)))
plt.plot([mE, 20000],[0.00001,17.5124867], linestyle='--', color='m', label="O(N4)")



plt.legend(('', '','Constante', 'O(N)', 'O(N2)', 'O(N3)', 'O(N4)'),
           loc='lower left')

plt.savefig("grafico_Matriz_llena_inv_E6.png")


plt.show()


-------------------------------------------------------------------------------------------------------------------------------------
-
-
-
-
-
-
-
-
-
-
-
-
-
-


# MATRIZ INVERTIDA DISPERSA

from numpy import double, ones, array
import scipy.sparse as sp
import scipy.sparse as sparse
import scipy.sparse.linalg as lin
from laplaciana import matriz_laplaciana
from time import perf_counter
import matplotlib.pylab as plt

Ns = [10, 20, 30, 50, 70, 100, 150, 200, 400, 600, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 7000, 10000, 20000]


Ms = []
dt = []

f = open("Matriz_dispersa_inv_E6.txt", "a")


for N in Ns:
    
    
    
    t1 = perf_counter()
    
    A = matriz_laplaciana(N, double)
    Acsc = sp.csc_matrix(A)
    
    t2 = perf_counter()
    
    Ainv = lin.inv(Acsc)
    #print (array(Ainv))
    
    t3 = perf_counter()

    dt_ensamblaje = t2 - t1
    Ms_solucion = t3 - t2
    
    dt.append(dt_ensamblaje)
    Ms.append(Ms_solucion)
    
    f.write(f"Matriz de {N}x{N}"+"\n")
    f.write(f"Tiempo ensamblaje: {dt_ensamblaje} s"+"\n")
    f.write(f"Tiempo solución: {Ms_solucion} s"+"\n")
    
    # print ("Matriz de ", N)
    # print(f"Tiempo ensamblaje: {dt_ensamblaje} s")
    # print(f"Tiempo solución: {dt_solucion} s")
    
    # print (f"Tiempo de ensamblaje: {t2-t1}")
    # print (f"Tiempo de solucion: {t3-t2}")
    
f.close()
    
    


# plt.figure(1)
plt.subplot(2, 1, 1)
plt.ylabel("Tiempo de Ensamblado")
plt.loglog(Ns, dt,"k", marker = "o")

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
plt.plot([10, 20000],[10**(3),10**(3)], linestyle='--', color='W', label="Constante" )

#LINEA AZUL (CONSTANTE)
plt.plot([10, 20000],[ 0.00369213, 0.00369213], linestyle='--', color='royalblue', label="Constante" )

# LINEA NARANJA (ON)
plt.plot([10, 20000],[0.00411,  0.00369213], linestyle='--', color='gold', label="O(N)")

# LINEA VERDE (ON2)
gE1 = (20000)*(((0.0000001/0.1878)**(1/2)))
plt.plot([gE1, 20000],[0.00001, 0.00369213], linestyle='--', color='g', label="O(N2)")

# LINEA ROJA (ON3)
rE1 = (20000)*(((0.00000001/ 0.00369213)**(1/3)))
plt.plot([rE1, 20000],[0.00001, 0.00369213], linestyle='--', color='r', label="O(N3)")

# LINEA MORADA (ON4)
mE1 = (20000)*(((0.00000001/ 0.00369213)**(1/4)))
plt.plot([mE1, 20000],[0.00001, 0.00369213], linestyle='--', color='m', label="O(N4)")



# plt.figure(2)
plt.subplot(2, 1, 2)
plt.ylabel("Tiempo de Solución ")
plt.xlabel("Tamaño Matriz N")
plt.loglog(Ns, Ms, "k", marker = "o")

#Yticks (Tiempo Transcurrido (s))
labelsY1 = ["0,1 ms", "1 ms", "10 ms", "0,1 s", "1 s", "10 s", "1 min", "10 min"]
labels_Y1 = [10**(-4), 10**(-3), 10**(-2), 10**(-1), 10**(0), 10**(1), 10**(2), 10**(3) ]
plt.yticks(labels_Y1, labelsY1)

#Xticks
labelsX = ["10", "20", "50", "100", "200", "500", "1000", "2000", "5000", "10000", "20000" ]
xlabel2 = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
plt.xticks(xlabel2, labelsX, rotation = 45)



#   LINEAS INTERMITENTES

#LINEA BLANCA 
plt.plot([10, 20000],[10**(3),10**(3)], linestyle='--', color='W', label="Constante" )

#LINEA AZUL (CONSTANTE)
plt.plot([10, 20000],[17.65211,17.65211], linestyle='--', color='royalblue', label="Constante" )

# LINEA NARANJA (ON)
plt.plot([10, 20000],[0.005383,17.65211], linestyle='--', color='gold', label="O(N)")

# LINEA VERDE (ON2)
gE = (20000)*(((0.00000001/17.65211)**(1/2)))
plt.plot([gE, 20000],[0.00001,17.65211], linestyle='--', color='g', label="O(N2)")

# LINEA ROJA (ON3)
rE = (20000)*(((0.00000001/17.65211)**(1/3)))
plt.plot([rE, 20000],[0.00001,17.65211], linestyle='--', color='r', label="O(N3)")

# LINEA MORADA (ON4)
mE = (20000)*(((0.00000001/17.65211)**(1/4)))
plt.plot([mE, 20000],[0.00001,17.65211], linestyle='--', color='m', label="O(N4)")


plt.legend(('', '','Constante', 'O(N)', 'O(N2)', 'O(N3)', 'O(N4)'),
           loc='lower left')

plt.savefig("grafico_Matriz_dispersa_inv_E6.png")


plt.show()


-------------------------------------------------------------------------------------------------------------------------




















