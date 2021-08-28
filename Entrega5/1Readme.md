
- Comente cómo esta elección se ve reflejada en el desempeño y complejidad algorítmica mostrada. 

- La elección de crear el codigo de la laplaciana fue de gran ayuda para los calculos sobre esta ya que el formato de su calculo se baso en obtener una diagonal de puros (2) y sus numeros a los dos lados serian (-1). Por lo que esto ayudo a su ensamblaje. 

- En cuanto a la resolucion A@B, fue tan eficienta la formacion de la laplaciana, que esta sirvio para poder correr el codigo muy rapido llegando hasta a correr numero grandes como matrices de 200.000 por 200.000 y sin niun probelema con una rapida respuesta. 
- Al hacerlo con la funcion "sparse.csr_matrix(A)", esta se agilizo considerablemente ya que esta no almacena los (0) de cada matriz, llegando a reducir el numero de complejos y largos calculos ahorando la informacion que no se iba a usar. 
- Para finalizar, me gustaria recalcar que la gran razon de la rapidez de desempeño de nuestro codigo se ve primero por el codigo de la laplaciana, y segundo, por el uso de la funcion "sparse.csr_matrix(A)". 


A continuación, dejare mi codigo comentado tal cual como se pidio:



```
"""


import numpy as np
from time import perf_counter
from numpy import half, single, double, longdouble, zeros, eye
from numpy import float16, float32, float64
from scipy.linalg import inv
from matriz_laplaciana import matriz_laplaciana
import matplotlib.pylab as plt
import scipy.sparse as sparse


N = 10

Ns = [10, 15, 20, 30, 40, 50, 60, 70, 80, 100, 130, 160, 200, 300, 400, 500, 650, 800, 900, 1000, 1500, 2000, 2500, 3500, 5000, 6000, 8000, 10000, 15000, 20000, 40000, 55000, 80000, 100000, 140000, 170000, 200000]
#Ns = [10,20,  50, 100, 200,300, 500,800 ,  1000, 1500]

Ms = []
dt = []

f = open("Matriz_llena_E5.txt", "a")

for N in Ns:
    
    t1 = t1 = perf_counter()
    A = matriz_laplaciana(N)    
    B = matriz_laplaciana(N)
    t2 = perf_counter()
    
    
    t3 = perf_counter()
    x = A@B
    #x = sp.matmul(A,B)

    t4 = perf_counter()
    
    # print (x)
    
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
plt.subplot(2, 1, 1)
plt.ylabel("Tiempo de Ensamblado")
plt.loglog(Ns, Ms, marker = "o")

#Yticks (Tiempo Transcurrido (s))
labelsY1 = ["0,1 ms", "1 ms", "10 ms", "0,1 s", "1 s", "10 s", "1 min", "10 min"]
labels_Y1 = [10**(-4), 10**(-3), 10**(-2), 10**(-1), 10**(0), 10**(1), 10**(2), 10**(3) ]
plt.yticks(labels_Y1, labelsY1)

#Xticks
labelsX = ["", "", "", "", "", "", "", "", "", "", "" ]
xlabel2 = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
plt.xticks(xlabel2, labelsX, rotation = 45)



# plt.figure(2)
plt.subplot(2, 1, 2)
plt.ylabel("Tiempo de Solución ")
plt.xlabel("Tamaño Matriz N")
plt.loglog(Ns, dt,  marker = "o")

#Yticks (Tiempo Transcurrido (s))
labelsY1 = ["0,1 ms", "1 ms", "10 ms", "0,1 s", "1 s", "10 s", "1 min", "10 min"]
labels_Y1 = [10**(-4), 10**(-3), 10**(-2), 10**(-1), 10**(0), 10**(1), 10**(2), 10**(3) ]
plt.yticks(labels_Y1, labelsY1)

#Xticks
labelsX = ["10", "20", "50", "100", "200", "500", "1000", "2000", "5000", "10000", "20000" ]
xlabel2 = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
plt.xticks(xlabel2, labelsX, rotation = 45)


plt.legend(('float', 'double'),
           loc='center left')




plt.savefig("grafico_Matriz_llena.png")


plt.show()


"""
```

-
-
-
-
-



Y el segundo codigo comentado.



```
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
plt.loglog(Ns, Ms, marker = "o")

#Yticks (Tiempo Transcurrido (s))
labelsY1 = ["0,1 ms", "1 ms", "10 ms", "0,1 s", "1 s", "10 s", "1 min", "10 min"]
labels_Y1 = [10**(-4), 10**(-3), 10**(-2), 10**(-1), 10**(0), 10**(1), 10**(2), 10**(3) ]
plt.yticks(labels_Y1, labelsY1)

#Xticks
labelsX = ["", "", "", "", "", "", "", "", "", "", "" ]
xlabel2 = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
plt.xticks(xlabel2, labelsX, rotation = 45)


# plt.figure(2)
plt.subplot(2, 1, 2)
plt.ylabel("Tiempo de Solución ")
plt.xlabel("Tamaño Matriz N")
plt.loglog(Ns, dt,  marker = "o")

#Yticks (Tiempo Transcurrido (s))
labelsY1 = ["0,1 ms", "1 ms", "10 ms", "0,1 s", "1 s", "10 s", "1 min", "10 min"]
labels_Y1 = [10**(-4), 10**(-3), 10**(-2), 10**(-1), 10**(0), 10**(1), 10**(2), 10**(3) ]
plt.yticks(labels_Y1, labelsY1)

#Xticks
labelsX = ["10", "20", "50", "100", "200", "500", "1000", "2000", "5000", "10000", "20000" ]
xlabel2 = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
plt.xticks(xlabel2, labelsX, rotation = 45)


plt.legend(('float', 'double'),
           loc='center left')




plt.savefig("grafico_Matriz_dispersa.png")


plt.show()


"""
```




