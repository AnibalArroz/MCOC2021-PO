
La entrega disponia de parte a) y parte b), y cada parte tenia su subparte (I, II, III, IV, etc). Ademas, en la parte b9, desde la subparte II a la V, tenian otras dos subsubpartes (False y True). 
por eso los codigos los nombre asi



Caso a) I

import scipy as sp
from time import perf_counter
from numpy import half, single, double, longdouble, zeros, eye
from numpy import float16, float32, float64
from scipy.linalg import inv
from laplaciana import laplaciana
import matplotlib.pylab as plt
from scipy.linalg import solve

N = 10000
Ns = [10, 15, 20, 30, 40, 50, 60, 70, 80, 100, 130, 160, 200, 300, 400, 500, 650, 800, 900, 1000, 1500, 2000, 2500, 3500, 5000, 6000, 8000, 10000]
#Ns = [10, 15, 20, 30, 40, 50, 60, 70, 80, 100, 130, 160]
#Ns = [4]
t0 = perf_counter()
dt = []

f = open("caso_A_I_float.txt", "a")

for N in Ns:
    temp = []
    for i in range(10):      
        t1 = perf_counter()
        A = laplaciana(N, dtype=float)
        Aml = inv(A)
        B = sp.ones(N)
        x = sp.matmul(Aml,B)
        t2 = perf_counter()
        dt_inversion = t2 - t1
        temp.append(dt_inversion)
    mean = sp.mean(temp)
    dt.append(mean)
        
    f.write(f"Matriz de {N}x{N}"+"\n")
    f.write(f"Tiempo promedio de solución: {dt_inversion} s"+"\n")
    f.write(f"Solución: {x} s"+"\n")
    #print ("Matriz de ", N)
    #print(f"Tiempo solución: {dt_inversion} s")

f.close()

Ks = Ns
dt1 = []

d = open("caso_A_I_double.txt", "a")

for N in Ns:
    temp2 = []
    for i in range(10):
        t4 = perf_counter()
        A = laplaciana(N, dtype=double)
        Aml = inv(A)
        B = sp.ones(N)
        x = sp.matmul(Aml,B)
        t5 = perf_counter()
        dt1_inversion = t5 - t4
        temp2.append(dt1_inversion)
    mean2 = sp.mean(temp2)
    dt1.append(mean2)
    
    
    d.write(f"Matriz de {N}x{N}"+"\n")
    d.write(f"Tiempo promedio de solución: {dt1_inversion} s"+"\n")
    d.write(f"Solución: {x} s"+"\n")
    
    #print ("Matriz de ", N)
    #print(f"Tiempo solución: {dt1_inversion} s")

d.close()

plt.loglog(Ks,dt1, marker = "o")
plt.ylabel("Tiempo Transcurrido (s)")
plt.xlabel("Tamaño Matiz N")
plt.loglog(Ns, dt,  marker = "o")

plt.legend(('float', 'double'),
           loc='upper right')

#Yticks (Tiempo Transcurrido (s))
labelsY1 = ["0,1 ms", "1 ms", "10 ms", "0,1 s", "1 s", "10 s", "1 min", "10 min"]
labels_Y1 = [10**(-4), 10**(-3), 10**(-2), 10**(-1), 10**(0), 10**(1), 10**(2), 10**(3) ]
plt.yticks(labels_Y1, labelsY1)

#Xticks
labelsX = ["10", "20", "50", "100", "200", "500", "1000", "2000", "5000", "10000", "20000" ]
xlabel2 = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
plt.xticks(xlabel2, labelsX, rotation = 45)

plt.grid(True)

plt.savefig("grafico_A_I.png")
plt.show()


-
-
-
-


Caso a) II


import scipy as sp
from time import perf_counter
from numpy import half, single, double, longdouble, zeros, eye
from numpy import float16, float32, float64
from scipy.linalg import inv
from laplaciana import laplaciana
import matplotlib.pylab as plt
from scipy.linalg import solve

N = 10000
Ns = [10, 15, 20, 30, 40, 50, 60, 70, 80, 100, 130, 160, 200, 300, 400, 500, 650, 800, 900, 1000, 1500, 2000, 2500, 3500, 5000, 7000, 10000]
#Ns = [10, 15, 20, 30, 40, 50, 60, 70, 80, 100, 130, 160]
#Ns = [4]
t0 = perf_counter()
dt = []

f = open("caso_A_II_float.txt", "a")

for N in Ns:
    temp = []
    for i in range(10):      
        t1 = perf_counter()
        A = laplaciana(N, dtype=float)
        B = sp.ones(N)
        x = solve(A, B)
        t2 = perf_counter()
        dt_inversion = t2 - t1
        temp.append(dt_inversion)
    mean = sp.mean(temp)
    dt.append(mean)
        
    f.write(f"Matriz de {N}x{N}"+"\n")
    f.write(f"Tiempo promedio de solución: {dt_inversion} s"+"\n")
    f.write(f"Solución: {x} s"+"\n")
    #print ("Matriz de ", N)
    #print(f"Tiempo solución: {dt_inversion} s")

f.close()

Ks = Ns
dt1 = []

d = open("caso_A_II_double.txt", "a")

for N in Ns:
    temp2 = []
    for i in range(10):
        t4 = perf_counter()
        A = laplaciana(N, dtype=double)
        B = sp.ones(N)
        x = solve(A, B)
        t5 = perf_counter()
        dt1_inversion = t5 - t4
        temp2.append(dt1_inversion)
    mean2 = sp.mean(temp2)
    dt1.append(mean2)
    
    
    d.write(f"Matriz de {N}x{N}"+"\n")
    d.write(f"Tiempo promedio de solución: {dt1_inversion} s"+"\n")
    d.write(f"Solución: {x} s"+"\n")
    
    #print ("Matriz de ", N)
    #print(f"Tiempo solución: {dt1_inversion} s")

d.close()

plt.loglog(Ks,dt1, marker = "o")
plt.ylabel("Tiempo Transcurrido (s)")
plt.xlabel("Tamaño Matiz N")
plt.loglog(Ns, dt,  marker = "o")

plt.legend(('float', 'double'),
           loc='upper right')

#Yticks (Tiempo Transcurrido (s))
labelsY1 = ["0,1 ms", "1 ms", "10 ms", "0,1 s", "1 s", "10 s", "1 min", "10 min"]
labels_Y1 = [10**(-4), 10**(-3), 10**(-2), 10**(-1), 10**(0), 10**(1), 10**(2), 10**(3) ]
plt.yticks(labels_Y1, labelsY1)

#Xticks
labelsX = ["10", "20", "50", "100", "200", "500", "1000", "2000", "5000", "10000", "20000" ]
xlabel2 = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
plt.xticks(xlabel2, labelsX, rotation = 45)

plt.grid(True)

plt.savefig("grafico_A_II.png")
plt.show()

-
-
-
-

Caso a) III


import scipy as sp
from time import perf_counter
from numpy import half, single, double, longdouble, zeros, eye
from numpy import float16, float32, float64
from scipy.linalg import inv
from laplaciana import laplaciana
import matplotlib.pylab as plt
from scipy.linalg import solve

N = 10000
Ns = [10, 15, 20, 30, 40, 50, 60, 70, 80, 100, 130, 160, 200, 300, 400, 500, 650, 800, 900, 1000, 1500, 2000, 2500, 3500, 5000, 7000, 10000]
#Ns = [10, 15, 20, 30, 40, 50, 60, 70, 80, 100, 130, 160]
#Ns = [4]
t0 = perf_counter()
dt = []

f = open("caso_A_III_float.txt", "a")

for N in Ns:
    temp = []
    for i in range(10):      
        t1 = perf_counter()
        A = laplaciana(N, dtype=float)
        B = sp.ones(N)
        x = solve(A, B, assume_a='pos')
        t2 = perf_counter()
        dt_inversion = t2 - t1
        temp.append(dt_inversion)
    mean = sp.mean(temp)
    dt.append(mean)
        
    f.write(f"Matriz de {N}x{N}"+"\n")
    f.write(f"Tiempo promedio de solución: {dt_inversion} s"+"\n")
    f.write(f"Solución: {x} s"+"\n")
    #print ("Matriz de ", N)
    #print(f"Tiempo solución: {dt_inversion} s")

f.close()

Ks = Ns
dt1 = []

d = open("caso_A_III_double.txt", "a")

for N in Ns:
    temp2 = []
    for i in range(10):
        t4 = perf_counter()
        A = laplaciana(N, dtype=double)
        B = sp.ones(N)
        x = solve(A, B, assume_a='pos')
        t5 = perf_counter()
        dt1_inversion = t5 - t4
        temp2.append(dt1_inversion)
    mean2 = sp.mean(temp2)
    dt1.append(mean2)
    
    
    d.write(f"Matriz de {N}x{N}"+"\n")
    d.write(f"Tiempo promedio de solución: {dt1_inversion} s"+"\n")
    d.write(f"Solución: {x} s"+"\n")
    
    #print ("Matriz de ", N)
    #print(f"Tiempo solución: {dt1_inversion} s")

d.close()

plt.loglog(Ks,dt1, marker = "o")
plt.ylabel("Tiempo Transcurrido (s)")
plt.xlabel("Tamaño Matiz N")
plt.loglog(Ns, dt,  marker = "o")

plt.legend(('float', 'double'),
           loc='upper right')

#Yticks (Tiempo Transcurrido (s))
labelsY1 = ["0,1 ms", "1 ms", "10 ms", "0,1 s", "1 s", "10 s", "1 min", "10 min"]
labels_Y1 = [10**(-4), 10**(-3), 10**(-2), 10**(-1), 10**(0), 10**(1), 10**(2), 10**(3) ]
plt.yticks(labels_Y1, labelsY1)

#Xticks
labelsX = ["10", "20", "50", "100", "200", "500", "1000", "2000", "5000", "10000", "20000" ]
xlabel2 = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
plt.xticks(xlabel2, labelsX, rotation = 45)

plt.grid(True)

plt.savefig("grafico_A_III.png")
plt.show()

-
-
-


Caso a) IV


import scipy as sp
from time import perf_counter
from numpy import half, single, double, longdouble, zeros, eye
from numpy import float16, float32, float64
from scipy.linalg import inv
from laplaciana import laplaciana
import matplotlib.pylab as plt
from scipy.linalg import solve

N = 10000
Ns = [10, 15, 20, 30, 40, 50, 60, 70, 80, 100, 130, 160, 200, 300, 400, 500, 650, 800, 900, 1000, 1500, 2000, 2500, 3500, 5000, 7000, 10000]
#Ns = [10, 15, 20, 30, 40, 50, 60, 70, 80, 100, 130, 160]
#Ns = [4]
t0 = perf_counter()
dt = []

f = open("caso_A_IV_float.txt", "a")

for N in Ns:
    temp = []
    for i in range(10):      
        t1 = perf_counter()
        A = laplaciana(N, dtype=float)
        B = sp.ones(N)
        x = solve(A, B, assume_a='sym')
        t2 = perf_counter()
        dt_inversion = t2 - t1
        temp.append(dt_inversion)
    mean = sp.mean(temp)
    dt.append(mean)
        
    f.write(f"Matriz de {N}x{N}"+"\n")
    f.write(f"Tiempo promedio de solución: {dt_inversion} s"+"\n")
    f.write(f"Solución: {x} s"+"\n")
    #print ("Matriz de ", N)
    #print(f"Tiempo solución: {dt_inversion} s")

f.close()

Ks = Ns
dt1 = []

d = open("caso_A_IV_double.txt", "a")

for N in Ns:
    temp2 = []
    for i in range(10):
        t4 = perf_counter()
        A = laplaciana(N, dtype=double)
        B = sp.ones(N)
        x = solve(A, B, assume_a='sym')
        t5 = perf_counter()
        dt1_inversion = t5 - t4
        temp2.append(dt1_inversion)
    mean2 = sp.mean(temp2)
    dt1.append(mean2)
    
    
    d.write(f"Matriz de {N}x{N}"+"\n")
    d.write(f"Tiempo promedio de solución: {dt1_inversion} s"+"\n")
    d.write(f"Solución: {x} s"+"\n")
    
    #print ("Matriz de ", N)
    #print(f"Tiempo solución: {dt1_inversion} s")

d.close()

plt.loglog(Ks,dt1, marker = "o")
plt.ylabel("Tiempo Transcurrido (s)")
plt.xlabel("Tamaño Matiz N")
plt.loglog(Ns, dt,  marker = "o")

plt.legend(('float', 'double'),
           loc='upper right')

#Yticks (Tiempo Transcurrido (s))
labelsY1 = ["0,1 ms", "1 ms", "10 ms", "0,1 s", "1 s", "10 s", "1 min", "10 min"]
labels_Y1 = [10**(-4), 10**(-3), 10**(-2), 10**(-1), 10**(0), 10**(1), 10**(2), 10**(3) ]
plt.yticks(labels_Y1, labelsY1)

#Xticks
labelsX = ["10", "20", "50", "100", "200", "500", "1000", "2000", "5000", "10000", "20000" ]
xlabel2 = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
plt.xticks(xlabel2, labelsX, rotation = 45)

plt.grid(True)

plt.savefig("grafico_A_IV.png")
plt.show()

-
-
-

Caso a) V


import scipy as sp
from time import perf_counter
from numpy import half, single, double, longdouble, zeros, eye
from numpy import float16, float32, float64
from scipy.linalg import inv
from laplaciana import laplaciana
import matplotlib.pylab as plt
from scipy.linalg import solve

N = 10000
Ns = [10, 15, 20, 30, 40, 50, 60, 70, 80, 100, 130, 160, 200, 300, 400, 500, 650, 800, 900, 1000, 1500, 2000, 2500, 3500, 5000, 7000, 10000]
#Ns = [10, 15, 20, 30, 40, 50, 60, 70, 80, 100, 130, 160]
#Ns = [4]
t0 = perf_counter()
dt = []

f = open("caso_A_V_float.txt", "a")

for N in Ns:
    temp = []
    for i in range(10):      
        t1 = perf_counter()
        A = laplaciana(N, dtype=float)
        B = sp.ones(N)
        x = solve(A, B, overwrite_a=True)
        t2 = perf_counter()
        dt_inversion = t2 - t1
        temp.append(dt_inversion)
    mean = sp.mean(temp)
    dt.append(mean)
        
    f.write(f"Matriz de {N}x{N}"+"\n")
    f.write(f"Tiempo promedio de solución: {dt_inversion} s"+"\n")
    f.write(f"Solución: {x} s"+"\n")
    #print ("Matriz de ", N)
    #print(f"Tiempo solución: {dt_inversion} s")

f.close()

Ks = Ns
dt1 = []

d = open("caso_A_V_double.txt", "a")

for N in Ns:
    temp2 = []
    for i in range(10):
        t4 = perf_counter()
        A = laplaciana(N, dtype=double)
        B = sp.ones(N)
        x = solve(A, B, overwrite_a=True)
        t5 = perf_counter()
        dt1_inversion = t5 - t4
        temp2.append(dt1_inversion)
    mean2 = sp.mean(temp2)
    dt1.append(mean2)
    
    
    d.write(f"Matriz de {N}x{N}"+"\n")
    d.write(f"Tiempo promedio de solución: {dt1_inversion} s"+"\n")
    d.write(f"Solución: {x} s"+"\n")
    
    #print ("Matriz de ", N)
    #print(f"Tiempo solución: {dt1_inversion} s")

d.close()

plt.loglog(Ks,dt1, marker = "o")
plt.ylabel("Tiempo Transcurrido (s)")
plt.xlabel("Tamaño Matiz N")
plt.loglog(Ns, dt,  marker = "o")

plt.legend(('float', 'double'),
           loc='upper right')

#Yticks (Tiempo Transcurrido (s))
labelsY1 = ["0,1 ms", "1 ms", "10 ms", "0,1 s", "1 s", "10 s", "1 min", "10 min"]
labels_Y1 = [10**(-4), 10**(-3), 10**(-2), 10**(-1), 10**(0), 10**(1), 10**(2), 10**(3) ]
plt.yticks(labels_Y1, labelsY1)

#Xticks
labelsX = ["10", "20", "50", "100", "200", "500", "1000", "2000", "5000", "10000", "20000" ]
xlabel2 = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
plt.xticks(xlabel2, labelsX, rotation = 45)

plt.grid(True)

plt.savefig("grafico_A_V.png")
plt.show()

-
-
-

Caso a) VI


import scipy as sp
from time import perf_counter
from numpy import half, single, double, longdouble, zeros, eye
from numpy import float16, float32, float64
from scipy.linalg import inv
from laplaciana import laplaciana
import matplotlib.pylab as plt
from scipy.linalg import solve

N = 10000
Ns = [10, 15, 20, 30, 40, 50, 60, 70, 80, 100, 130, 160, 200, 300, 400, 500, 650, 800, 900, 1000, 1500, 2000, 2500, 3500, 5000, 7000, 10000]
#Ns = [10, 15, 20, 30, 40, 50, 60, 70, 80, 100, 130, 160]
#Ns = [4]
t0 = perf_counter()
dt = []

f = open("caso_A_VI_float.txt", "a")

for N in Ns:
    temp = []
    for i in range(10):      
        t1 = perf_counter()
        A = laplaciana(N, dtype=float)
        B = sp.ones(N)
        x = solve(A, B, overwrite_b=True)
        t2 = perf_counter()
        dt_inversion = t2 - t1
        temp.append(dt_inversion)
    mean = sp.mean(temp)
    dt.append(mean)
        
    f.write(f"Matriz de {N}x{N}"+"\n")
    f.write(f"Tiempo promedio de solución: {dt_inversion} s"+"\n")
    f.write(f"Solución: {x} s"+"\n")
    #print ("Matriz de ", N)
    #print(f"Tiempo solución: {dt_inversion} s")

f.close()

Ks = Ns
dt1 = []

d = open("caso_A_VI_double.txt", "a")

for N in Ns:
    temp2 = []
    for i in range(10):
        t4 = perf_counter()
        A = laplaciana(N, dtype=double)
        B = sp.ones(N)
        x = solve(A, B, overwrite_b=True)
        t5 = perf_counter()
        dt1_inversion = t5 - t4
        temp2.append(dt1_inversion)
    mean2 = sp.mean(temp2)
    dt1.append(mean2)
    
    
    d.write(f"Matriz de {N}x{N}"+"\n")
    d.write(f"Tiempo promedio de solución: {dt1_inversion} s"+"\n")
    d.write(f"Solución: {x} s"+"\n")

    
    #print ("Matriz de ", N)
    #print(f"Tiempo solución: {dt1_inversion} s")

d.close()

plt.loglog(Ks,dt1, marker = "o")
plt.ylabel("Tiempo Transcurrido (s)")
plt.xlabel("Tamaño Matiz N")
plt.loglog(Ns, dt,  marker = "o")

plt.legend(('float', 'double'),
           loc='upper right')

#Yticks (Tiempo Transcurrido (s))
labelsY1 = ["0,1 ms", "1 ms", "10 ms", "0,1 s", "1 s", "10 s", "1 min", "10 min"]
labels_Y1 = [10**(-4), 10**(-3), 10**(-2), 10**(-1), 10**(0), 10**(1), 10**(2), 10**(3) ]
plt.yticks(labels_Y1, labelsY1)

#Xticks
labelsX = ["10", "20", "50", "100", "200", "500", "1000", "2000", "5000", "10000", "20000" ]
xlabel2 = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
plt.xticks(xlabel2, labelsX, rotation = 45)

plt.grid(True)

plt.savefig("grafico_A_VI.png")
plt.show()

-
-
-

Caso a) VII


import scipy as sp
from time import perf_counter
from numpy import half, single, double, longdouble, zeros, eye
from numpy import float16, float32, float64
from scipy.linalg import inv
from laplaciana import laplaciana
import matplotlib.pylab as plt
from scipy.linalg import solve

N = 10000
Ns = [10, 15, 20, 30, 40, 50, 60, 70, 80, 100, 130, 160, 200, 300, 400, 500, 650, 800, 900, 1000, 1500, 2000, 2500, 3500, 5000, 7000, 10000]
#Ns = [10, 15, 20, 30, 40, 50, 60, 70, 80, 100, 130, 160]
#Ns = [4]
t0 = perf_counter()
dt = []

f = open("caso_A_VII_float.txt", "a")

for N in Ns:
    temp = []
    for i in range(10):      
        t1 = perf_counter()
        A = laplaciana(N, dtype=float)
        B = sp.ones(N)
        x = solve(A, B, overwrite_a=True, overwrite_b=True)
        t2 = perf_counter()
        dt_inversion = t2 - t1
        temp.append(dt_inversion)
    mean = sp.mean(temp)
    dt.append(mean)
        
    f.write(f"Matriz de {N}x{N}"+"\n")
    f.write(f"Tiempo promedio de solución: {dt_inversion} s"+"\n")
    f.write(f"Solución: {x} s"+"\n")
    #print ("Matriz de ", N)
    #print(f"Tiempo solución: {dt_inversion} s")

f.close()

Ks = Ns
dt1 = []

d = open("caso_A_VII_double.txt", "a")

for N in Ns:
    temp2 = []
    for i in range(10):
        t4 = perf_counter()
        A = laplaciana(N, dtype=double)
        B = sp.ones(N)
        x = solve(A, B, overwrite_a=True, overwrite_b=True)
        t5 = perf_counter()
        dt1_inversion = t5 - t4
        temp2.append(dt1_inversion)
    mean2 = sp.mean(temp2)
    dt1.append(mean2)
    
    
    d.write(f"Matriz de {N}x{N}"+"\n")
    d.write(f"Tiempo promedio de solución: {dt1_inversion} s"+"\n")
    d.write(f"Solución: {x} s"+"\n")
    
    #print ("Matriz de ", N)
    #print(f"Tiempo solución: {dt1_inversion} s")

d.close()

plt.loglog(Ks,dt1, marker = "o")
plt.ylabel("Tiempo Transcurrido (s)")
plt.xlabel("Tamaño Matiz N")
plt.loglog(Ns, dt,  marker = "o")

plt.legend(('float', 'double'),
           loc='upper right')

#Yticks (Tiempo Transcurrido (s))
labelsY1 = ["0,1 ms", "1 ms", "10 ms", "0,1 s", "1 s", "10 s", "1 min", "10 min"]
labels_Y1 = [10**(-4), 10**(-3), 10**(-2), 10**(-1), 10**(0), 10**(1), 10**(2), 10**(3) ]
plt.yticks(labels_Y1, labelsY1)

#Xticks
labelsX = ["10", "20", "50", "100", "200", "500", "1000", "2000", "5000", "10000", "20000" ]
xlabel2 = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
plt.xticks(xlabel2, labelsX, rotation = 45)

plt.grid(True)

plt.savefig("grafico_A_VII.png")
plt.show()

-
-
-


Caso b) I


import scipy as sp
from time import perf_counter
from numpy import half, single, double, longdouble, zeros, eye
from numpy import float16, float32, float64
from scipy.linalg import inv
from laplaciana import laplaciana
import matplotlib.pylab as plt
from scipy.linalg import eigh

N = 10000
Ns = [10, 15, 20, 30, 40, 50, 60, 70, 80, 100, 130, 160, 200, 300, 400, 500, 650, 800, 900, 1000, 1500, 2000, 2500, 3500, 5000, 7000, 10000]
#Ns = [10, 15, 20, 30, 40, 50, 60, 70, 80, 100, 130, 160]
#Ns = [4]
t0 = perf_counter()
dt = []

f = open("caso_B_I_float.txt", "a")

for N in Ns:
    temp = []
    for i in range(10):      
        t1 = perf_counter()
        A = laplaciana(N, dtype=float)
        x = eigh(A)
        t2 = perf_counter()
        dt_inversion = t2 - t1
        temp.append(dt_inversion)
    mean = sp.mean(temp)
    dt.append(mean)
        
    f.write(f"Matriz de {N}x{N}"+"\n")
    f.write(f"Tiempo promedio de solución: {dt_inversion} s"+"\n")
    f.write(f"Solución: {x} s"+"\n")
    #print ("Matriz de ", N)
    #print(f"Tiempo solución: {dt_inversion} s")

f.close()

Ks = Ns
dt1 = []

d = open("caso_B_I_double.txt", "a")

for N in Ns:
    temp2 = []
    for i in range(10):
        t4 = perf_counter()
        A = laplaciana(N, dtype=double)
        x = eigh(A)
        t5 = perf_counter()
        dt1_inversion = t5 - t4
        temp2.append(dt1_inversion)
    mean2 = sp.mean(temp2)
    dt1.append(mean2)
    
    
    d.write(f"Matriz de {N}x{N}"+"\n")
    d.write(f"Tiempo promedio de solución: {dt1_inversion} s"+"\n")
    d.write(f"Solución: {x} s"+"\n")
    
    #print ("Matriz de ", N)
    #print(f"Tiempo solución: {dt1_inversion} s")

d.close()

plt.loglog(Ks,dt1, marker = "o")
plt.ylabel("Tiempo Transcurrido (s)")
plt.xlabel("Tamaño Matiz N")
plt.loglog(Ns, dt,  marker = "o")

plt.legend(('float', 'double'),
           loc='upper right')

#Yticks (Tiempo Transcurrido (s))
labelsY1 = ["0,1 ms", "1 ms", "10 ms", "0,1 s", "1 s", "10 s", "1 min", "10 min"]
labels_Y1 = [10**(-4), 10**(-3), 10**(-2), 10**(-1), 10**(0), 10**(1), 10**(2), 10**(3) ]
plt.yticks(labels_Y1, labelsY1)

#Xticks
labelsX = ["10", "20", "50", "100", "200", "500", "1000", "2000", "5000", "10000", "20000" ]
xlabel2 = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
plt.xticks(xlabel2, labelsX, rotation = 45)

plt.grid(True)

plt.savefig("grafico_B_I.png")
plt.show()

-
-
-

Caso b) II caso False


import scipy as sp
from time import perf_counter
from numpy import half, single, double, longdouble, zeros, eye
from numpy import float16, float32, float64
from scipy.linalg import inv
from laplaciana import laplaciana
import matplotlib.pylab as plt
from scipy.linalg import eigh

N = 10000
Ns = [10, 15, 20, 30, 40, 50, 60, 70, 80, 100, 130, 160, 200, 300, 400, 500, 650, 900, 1000, 5000, 10000]
#Ns = [10, 15, 20, 30, 40, 50, 60, 70, 80, 100, 130, 160]
#Ns = [4]
t0 = perf_counter()
dt = []

f = open("caso_B_II_float_False.txt", "a")

for N in Ns:
    temp = []
    for i in range(10):      
        t1 = perf_counter()
        A = laplaciana(N, dtype=float)
        x = eigh(A, turbo="ev", overwrite_a=False)
        t2 = perf_counter()
        dt_inversion = t2 - t1
        temp.append(dt_inversion)
    mean = sp.mean(temp)
    dt.append(mean)
        
    f.write(f"Matriz de {N}x{N}"+"\n")
    f.write(f"Tiempo promedio de solución: {dt_inversion} s"+"\n")
    f.write(f"Solución: {x} s"+"\n")
    #print ("Matriz de ", N)
    #print(f"Tiempo solución: {dt_inversion} s")

f.close()

Ks = Ns
dt1 = []

d = open("caso_B_II_double_False.txt", "a")

for N in Ns:
    temp2 = []
    for i in range(10):
        t4 = perf_counter()
        A = laplaciana(N, dtype=double)
        x = eigh(A, turbo="ev", overwrite_a=False)
        t5 = perf_counter()
        dt1_inversion = t5 - t4
        temp2.append(dt1_inversion)
    mean2 = sp.mean(temp2)
    dt1.append(mean2)
    
    
    d.write(f"Matriz de {N}x{N}"+"\n")
    d.write(f"Tiempo promedio de solución: {dt1_inversion} s"+"\n")
    d.write(f"Solución: {x} s"+"\n")
    
    #print ("Matriz de ", N)
    #print(f"Tiempo solución: {dt1_inversion} s")

d.close()

plt.loglog(Ks,dt1, marker = "o")
plt.ylabel("Tiempo Transcurrido (s)")
plt.xlabel("Tamaño Matiz N")
plt.loglog(Ns, dt,  marker = "o")

plt.legend(('float', 'double'),
           loc='upper right')

#Yticks (Tiempo Transcurrido (s))
labelsY1 = ["0,1 ms", "1 ms", "10 ms", "0,1 s", "1 s", "10 s", "1 min", "10 min"]
labels_Y1 = [10**(-4), 10**(-3), 10**(-2), 10**(-1), 10**(0), 10**(1), 10**(2), 10**(3) ]
plt.yticks(labels_Y1, labelsY1)

#Xticks
labelsX = ["10", "20", "50", "100", "200", "500", "1000", "2000", "5000", "10000", "20000" ]
xlabel2 = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
plt.xticks(xlabel2, labelsX, rotation = 45)

plt.grid(True)

plt.savefig("grafico_B_II_False.png")
plt.show()

-
-
-

Caso b) II caso True


import scipy as sp
from time import perf_counter
from numpy import half, single, double, longdouble, zeros, eye
from numpy import float16, float32, float64
from scipy.linalg import inv
from laplaciana import laplaciana
import matplotlib.pylab as plt
from scipy.linalg import eigh

N = 10000
Ns = [10, 15, 20, 30, 40, 50, 60, 70, 80, 100, 130, 160, 200, 300, 400, 500, 650, 900, 1000, 5000]
#Ns = [10, 15, 20, 30, 40, 50, 60, 70, 80, 100, 130, 160]
#Ns = [4]
t0 = perf_counter()
dt = []

f = open("caso_B_II_float_True.txt", "a")

for N in Ns:
    temp = []
    for i in range(10):      
        t1 = perf_counter()
        A = laplaciana(N, dtype=float)
        x = eigh(A, turbo="ev", overwrite_a=True)
        t2 = perf_counter()
        dt_inversion = t2 - t1
        temp.append(dt_inversion)
    mean = sp.mean(temp)
    dt.append(mean)
        
    f.write(f"Matriz de {N}x{N}"+"\n")
    f.write(f"Tiempo promedio de solución: {dt_inversion} s"+"\n")
    f.write(f"Solución: {x} s"+"\n")
    #print ("Matriz de ", N)
    #print(f"Tiempo solución: {dt_inversion} s")

f.close()

Ks = Ns
dt1 = []

d = open("caso_B_II_double_True.txt", "a")

for N in Ns:
    temp2 = []
    for i in range(10):
        t4 = perf_counter()
        A = laplaciana(N, dtype=double)
        x = eigh(A, turbo="ev", overwrite_a=True)
        t5 = perf_counter()
        dt1_inversion = t5 - t4
        temp2.append(dt1_inversion)
    mean2 = sp.mean(temp2)
    dt1.append(mean2)
    
    
    d.write(f"Matriz de {N}x{N}"+"\n")
    d.write(f"Tiempo promedio de solución: {dt1_inversion} s"+"\n")
    d.write(f"Solución: {x} s"+"\n")
    
    #print ("Matriz de ", N)
    #print(f"Tiempo solución: {dt1_inversion} s")

d.close()

plt.loglog(Ks,dt1, marker = "o")
plt.ylabel("Tiempo Transcurrido (s)")
plt.xlabel("Tamaño Matiz N")
plt.loglog(Ns, dt,  marker = "o")

plt.legend(('float', 'double'),
           loc='upper right')

#Yticks (Tiempo Transcurrido (s))
labelsY1 = ["0,1 ms", "1 ms", "10 ms", "0,1 s", "1 s", "10 s", "1 min", "10 min"]
labels_Y1 = [10**(-4), 10**(-3), 10**(-2), 10**(-1), 10**(0), 10**(1), 10**(2), 10**(3) ]
plt.yticks(labels_Y1, labelsY1)

#Xticks
labelsX = ["10", "20", "50", "100", "200", "500", "1000", "2000", "5000", "10000", "20000" ]
xlabel2 = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
plt.xticks(xlabel2, labelsX, rotation = 45)

plt.grid(True)

plt.savefig("grafico_B_II_True.png")
plt.show()

-
-
-

Caso b) III caso False


import scipy as sp
from time import perf_counter
from numpy import half, single, double, longdouble, zeros, eye
from numpy import float16, float32, float64
from scipy.linalg import inv
from laplaciana import laplaciana
import matplotlib.pylab as plt
from scipy.linalg import eigh

N = 10000
Ns = [10, 15, 20, 30, 40, 50, 60, 70, 80, 100, 130, 160, 200, 300, 400, 500, 650, 900, 1000, 5000]
#Ns = [10, 15, 20, 30, 40, 50, 60, 70, 80, 100, 130, 160]
#Ns = [4]
t0 = perf_counter()
dt = []

f = open("caso_B_III_float_False.txt", "a")

for N in Ns:
    temp = []
    for i in range(10):      
        t1 = perf_counter()
        A = laplaciana(N, dtype=float)
        x = eigh(A, turbo="evd", overwrite_a=False)
        t2 = perf_counter()
        dt_inversion = t2 - t1
        temp.append(dt_inversion)
    mean = sp.mean(temp)
    dt.append(mean)
        
    f.write(f"Matriz de {N}x{N}"+"\n")
    f.write(f"Tiempo promedio de solución: {dt_inversion} s"+"\n")
    f.write(f"Solución: {x} s"+"\n")
    #print ("Matriz de ", N)
    #print(f"Tiempo solución: {dt_inversion} s")

f.close()

Ks = Ns
dt1 = []

d = open("caso_B_III_double_False.txt", "a")

for N in Ns:
    temp2 = []
    for i in range(10):
        t4 = perf_counter()
        A = laplaciana(N, dtype= double)
        x = eigh(A, turbo="evd", overwrite_a=False)
        t5 = perf_counter()
        dt1_inversion = t5 - t4
        temp2.append(dt1_inversion)
    mean2 = sp.mean(temp2)
    dt1.append(mean2)
    
    
    d.write(f"Matriz de {N}x{N}"+"\n")
    d.write(f"Tiempo promedio de solución: {dt1_inversion} s"+"\n")
    d.write(f"Solución: {x} s"+"\n")
    
    #print ("Matriz de ", N)
    #print(f"Tiempo solución: {dt1_inversion} s")

d.close()

plt.loglog(Ks,dt1, marker = "o")
plt.ylabel("Tiempo Transcurrido (s)")
plt.xlabel("Tamaño Matiz N")
plt.loglog(Ns, dt,  marker = "o")

plt.legend(('float', 'double'),
           loc='upper right')

#Yticks (Tiempo Transcurrido (s))
labelsY1 = ["0,1 ms", "1 ms", "10 ms", "0,1 s", "1 s", "10 s", "1 min", "10 min"]
labels_Y1 = [10**(-4), 10**(-3), 10**(-2), 10**(-1), 10**(0), 10**(1), 10**(2), 10**(3) ]
plt.yticks(labels_Y1, labelsY1)

#Xticks
labelsX = ["10", "20", "50", "100", "200", "500", "1000", "2000", "5000", "10000", "20000" ]
xlabel2 = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
plt.xticks(xlabel2, labelsX, rotation = 45)

plt.grid(True)

plt.savefig("grafico_B_III_False.png")
plt.show()

-
-
-

Caso b) III caso True


import scipy as sp
from time import perf_counter
from numpy import half, single, double, longdouble, zeros, eye
from numpy import float16, float32, float64
from scipy.linalg import inv
from laplaciana import laplaciana
import matplotlib.pylab as plt
from scipy.linalg import eigh

N = 10000
Ns = [10, 15, 20, 30, 40, 50, 60, 70, 80, 100, 130, 160, 200, 300, 400, 500, 650, 900, 1000]
#Ns = [10, 15, 20, 30, 40, 50, 60, 70, 80, 100, 130, 160]
#Ns = [4]
t0 = perf_counter()
dt = []

f = open("caso_B_III_float_True.txt", "a")

for N in Ns:
    temp = []
    for i in range(10):      
        t1 = perf_counter()
        A = laplaciana(N, dtype=float)
        x = eigh(A, turbo="evd", overwrite_a=True)
        t2 = perf_counter()
        dt_inversion = t2 - t1
        temp.append(dt_inversion)
    mean = sp.mean(temp)
    dt.append(mean)
        
    f.write(f"Matriz de {N}x{N}"+"\n")
    f.write(f"Tiempo promedio de solución: {dt_inversion} s"+"\n")
    f.write(f"Solución: {x} s"+"\n")
    #print ("Matriz de ", N)
    #print(f"Tiempo solución: {dt_inversion} s")

f.close()

Ks = Ns
dt1 = []

d = open("caso_B_III_double_True.txt", "a")

for N in Ns:
    temp2 = []
    for i in range(10):
        t4 = perf_counter()
        A = laplaciana(N, dtype= double)
        x = eigh(A, turbo="evd", overwrite_a=True)
        t5 = perf_counter()
        dt1_inversion = t5 - t4
        temp2.append(dt1_inversion)
    mean2 = sp.mean(temp2)
    dt1.append(mean2)
    
    
    d.write(f"Matriz de {N}x{N}"+"\n")
    d.write(f"Tiempo promedio de solución: {dt1_inversion} s"+"\n")
    d.write(f"Solución: {x} s"+"\n")
    
    #print ("Matriz de ", N)
    #print(f"Tiempo solución: {dt1_inversion} s")

d.close()

plt.loglog(Ks,dt1, marker = "o")
plt.ylabel("Tiempo Transcurrido (s)")
plt.xlabel("Tamaño Matiz N")
plt.loglog(Ns, dt,  marker = "o")

plt.legend(('float', 'double'),
           loc='upper right')

#Yticks (Tiempo Transcurrido (s))
labelsY1 = ["0,1 ms", "1 ms", "10 ms", "0,1 s", "1 s", "10 s", "1 min", "10 min"]
labels_Y1 = [10**(-4), 10**(-3), 10**(-2), 10**(-1), 10**(0), 10**(1), 10**(2), 10**(3) ]
plt.yticks(labels_Y1, labelsY1)

#Xticks
labelsX = ["10", "20", "50", "100", "200", "500", "1000", "2000", "5000", "10000", "20000" ]
xlabel2 = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
plt.xticks(xlabel2, labelsX, rotation = 45)

plt.grid(True)

plt.savefig("grafico_B_III_True.png")
plt.show()

-
-
-

Caso b) IV caso False


import scipy as sp
from time import perf_counter
from numpy import half, single, double, longdouble, zeros, eye
from numpy import float16, float32, float64
from scipy.linalg import inv
from laplaciana import laplaciana
import matplotlib.pylab as plt
from scipy.linalg import eigh

N = 10000
Ns = [10, 15, 20, 30, 40, 50, 60, 70, 80, 100, 130, 160, 200, 300, 400, 500, 650, 900, 1000]
#Ns = [10, 15, 20, 30, 40, 50, 60, 70, 80, 100, 130, 160]
#Ns = [4]
t0 = perf_counter()
dt = []

f = open("caso_B_IV_float_False.txt", "a")

for N in Ns:
    temp = []
    for i in range(10):      
        t1 = perf_counter()
        A = laplaciana(N, dtype=float)
        x = eigh(A, turbo="evr", overwrite_a=False)
        t2 = perf_counter()
        dt_inversion = t2 - t1
        temp.append(dt_inversion)
    mean = sp.mean(temp)
    dt.append(mean)
        
    f.write(f"Matriz de {N}x{N}"+"\n")
    f.write(f"Tiempo promedio de solución: {dt_inversion} s"+"\n")
    f.write(f"Solución: {x} s"+"\n")
    #print ("Matriz de ", N)
    #print(f"Tiempo solución: {dt_inversion} s")

f.close()

Ks = Ns
dt1 = []

d = open("caso_B_IV_double_False.txt", "a")

for N in Ns:
    temp2 = []
    for i in range(10):
        t4 = perf_counter()
        A = laplaciana(N, dtype= double)
        x = eigh(A, turbo="evr", overwrite_a=False)
        t5 = perf_counter()
        dt1_inversion = t5 - t4
        temp2.append(dt1_inversion)
    mean2 = sp.mean(temp2)
    dt1.append(mean2)
    
    
    d.write(f"Matriz de {N}x{N}"+"\n")
    d.write(f"Tiempo promedio de solución: {dt1_inversion} s"+"\n")
    d.write(f"Solución: {x} s"+"\n")
    
    #print ("Matriz de ", N)
    #print(f"Tiempo solución: {dt1_inversion} s")

d.close()

plt.loglog(Ks,dt1, marker = "o")
plt.ylabel("Tiempo Transcurrido (s)")
plt.xlabel("Tamaño Matiz N")
plt.loglog(Ns, dt,  marker = "o")

plt.legend(('float', 'double'),
           loc='upper right')

#Yticks (Tiempo Transcurrido (s))
labelsY1 = ["0,1 ms", "1 ms", "10 ms", "0,1 s", "1 s", "10 s", "1 min", "10 min"]
labels_Y1 = [10**(-4), 10**(-3), 10**(-2), 10**(-1), 10**(0), 10**(1), 10**(2), 10**(3) ]
plt.yticks(labels_Y1, labelsY1)

#Xticks
labelsX = ["10", "20", "50", "100", "200", "500", "1000", "2000", "5000", "10000", "20000" ]
xlabel2 = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
plt.xticks(xlabel2, labelsX, rotation = 45)

plt.grid(True)

plt.savefig("grafico_B_IV_False.png")
plt.show()

-
-
-

Caso b) IV caso True


import scipy as sp
from time import perf_counter
from numpy import half, single, double, longdouble, zeros, eye
from numpy import float16, float32, float64
from scipy.linalg import inv
from laplaciana import laplaciana
import matplotlib.pylab as plt
from scipy.linalg import eigh

N = 10000
Ns = [10, 15, 20, 30, 40, 50, 60, 70, 80, 100, 130, 160, 200, 300, 400, 500, 650, 900, 1000]
#Ns = [10, 15, 20, 30, 40, 50, 60, 70, 80, 100, 130, 160]
#Ns = [4]
t0 = perf_counter()
dt = []

f = open("caso_B_IV_float_True.txt", "a")

for N in Ns:
    temp = []
    for i in range(10):      
        t1 = perf_counter()
        A = laplaciana(N, dtype=float)
        x = eigh(A, turbo="evr", overwrite_a=True)
        t2 = perf_counter()
        dt_inversion = t2 - t1
        temp.append(dt_inversion)
    mean = sp.mean(temp)
    dt.append(mean)
        
    f.write(f"Matriz de {N}x{N}"+"\n")
    f.write(f"Tiempo promedio de solución: {dt_inversion} s"+"\n")
    f.write(f"Solución: {x} s"+"\n")
    #print ("Matriz de ", N)
    #print(f"Tiempo solución: {dt_inversion} s")

f.close()

Ks = Ns
dt1 = []

d = open("caso_B_IV_double_True.txt", "a")

for N in Ns:
    temp2 = []
    for i in range(10):
        t4 = perf_counter()
        A = laplaciana(N, dtype= double)
        x = eigh(A, turbo="evr", overwrite_a=True)
        t5 = perf_counter()
        dt1_inversion = t5 - t4
        temp2.append(dt1_inversion)
    mean2 = sp.mean(temp2)
    dt1.append(mean2)
    
    
    d.write(f"Matriz de {N}x{N}"+"\n")
    d.write(f"Tiempo promedio de solución: {dt1_inversion} s"+"\n")
    d.write(f"Solución: {x} s"+"\n")
    
    #print ("Matriz de ", N)
    #print(f"Tiempo solución: {dt1_inversion} s")

d.close()

plt.loglog(Ks,dt1, marker = "o")
plt.ylabel("Tiempo Transcurrido (s)")
plt.xlabel("Tamaño Matiz N")
plt.loglog(Ns, dt,  marker = "o")

plt.legend(('float', 'double'),
           loc='upper right')

#Yticks (Tiempo Transcurrido (s))
labelsY1 = ["0,1 ms", "1 ms", "10 ms", "0,1 s", "1 s", "10 s", "1 min", "10 min"]
labels_Y1 = [10**(-4), 10**(-3), 10**(-2), 10**(-1), 10**(0), 10**(1), 10**(2), 10**(3) ]
plt.yticks(labels_Y1, labelsY1)

#Xticks
labelsX = ["10", "20", "50", "100", "200", "500", "1000", "2000", "5000", "10000", "20000" ]
xlabel2 = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
plt.xticks(xlabel2, labelsX, rotation = 45)

plt.grid(True)

plt.savefig("grafico_B_IV_True.png")
plt.show()

-
-
-

Caso b) V caso False


import scipy as sp
from time import perf_counter
from numpy import half, single, double, longdouble, zeros, eye
from numpy import float16, float32, float64
from scipy.linalg import inv
from laplaciana import laplaciana
import matplotlib.pylab as plt
from scipy.linalg import eigh

N = 10000
Ns = [10, 15, 20, 30, 40, 50, 60, 70, 80, 100, 130, 160, 200, 300, 400, 500, 650, 900, 1000]
#Ns = [10, 15, 20, 30, 40, 50, 60, 70, 80, 100, 130, 160]
#Ns = [4]
t0 = perf_counter()
dt = []

f = open("caso_B_V_float_False.txt", "a")

for N in Ns:
    temp = []
    for i in range(10):      
        t1 = perf_counter()
        A = laplaciana(N, dtype=float)
        x = eigh(A, turbo="evx", overwrite_a=False)
        t2 = perf_counter()
        dt_inversion = t2 - t1
        temp.append(dt_inversion)
    mean = sp.mean(temp)
    dt.append(mean)
        
    f.write(f"Matriz de {N}x{N}"+"\n")
    f.write(f"Tiempo promedio de solución: {dt_inversion} s"+"\n")
    f.write(f"Solución: {x} s"+"\n")
    #print ("Matriz de ", N)
    #print(f"Tiempo solución: {dt_inversion} s")

f.close()

Ks = Ns
dt1 = []

d = open("caso_B_V_double_False.txt", "a")

for N in Ns:
    temp2 = []
    for i in range(10):
        t4 = perf_counter()
        A = laplaciana(N, dtype= double)
        x = eigh(A, turbo="evx", overwrite_a=False)
        t5 = perf_counter()
        dt1_inversion = t5 - t4
        temp2.append(dt1_inversion)
    mean2 = sp.mean(temp2)
    dt1.append(mean2)
    
    
    d.write(f"Matriz de {N}x{N}"+"\n")
    d.write(f"Tiempo promedio de solución: {dt1_inversion} s"+"\n")
    d.write(f"Solución: {x} s"+"\n")
    
    #print ("Matriz de ", N)
    #print(f"Tiempo solución: {dt1_inversion} s")

d.close()

plt.loglog(Ks,dt1, marker = "o")
plt.ylabel("Tiempo Transcurrido (s)")
plt.xlabel("Tamaño Matiz N")
plt.loglog(Ns, dt,  marker = "o")

plt.legend(('float', 'double'),
           loc='upper right')

#Yticks (Tiempo Transcurrido (s))
labelsY1 = ["0,1 ms", "1 ms", "10 ms", "0,1 s", "1 s", "10 s", "1 min", "10 min"]
labels_Y1 = [10**(-4), 10**(-3), 10**(-2), 10**(-1), 10**(0), 10**(1), 10**(2), 10**(3) ]
plt.yticks(labels_Y1, labelsY1)

#Xticks
labelsX = ["10", "20", "50", "100", "200", "500", "1000", "2000", "5000", "10000", "20000" ]
xlabel2 = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
plt.xticks(xlabel2, labelsX, rotation = 45)

plt.grid(True)

plt.savefig("grafico_B_V_False.png")
plt.show()

-
-
-
Caso b) V caso True


import scipy as sp
from time import perf_counter
from numpy import half, single, double, longdouble, zeros, eye
from numpy import float16, float32, float64
from scipy.linalg import inv
from laplaciana import laplaciana
import matplotlib.pylab as plt
from scipy.linalg import eigh

N = 10000
Ns = [10, 15, 20, 30, 40, 50, 60, 70, 80, 100, 130, 160, 200, 300, 400, 500, 650, 900, 1000]
#Ns = [10, 15, 20, 30, 40, 50, 60, 70, 80, 100, 130, 160]
#Ns = [4]
t0 = perf_counter()
dt = []

f = open("caso_B_V_float_True.txt", "a")

for N in Ns:
    temp = []
    for i in range(10):      
        t1 = perf_counter()
        A = laplaciana(N, dtype=float)
        x = eigh(A, turbo="evx", overwrite_a=True)
        t2 = perf_counter()
        dt_inversion = t2 - t1
        temp.append(dt_inversion)
    mean = sp.mean(temp)
    dt.append(mean)
        
    f.write(f"Matriz de {N}x{N}"+"\n")
    f.write(f"Tiempo promedio de solución: {dt_inversion} s"+"\n")
    f.write(f"Solución: {x} s"+"\n")
    #print ("Matriz de ", N)
    #print(f"Tiempo solución: {dt_inversion} s")

f.close()

Ks = Ns
dt1 = []

d = open("caso_B_V_double_True.txt", "a")

for N in Ns:
    temp2 = []
    for i in range(10):
        t4 = perf_counter()
        A = laplaciana(N, dtype= double)
        x = eigh(A, turbo="evx", overwrite_a=True)
        t5 = perf_counter()
        dt1_inversion = t5 - t4
        temp2.append(dt1_inversion)
    mean2 = sp.mean(temp2)
    dt1.append(mean2)
    
    
    d.write(f"Matriz de {N}x{N}"+"\n")
    d.write(f"Tiempo promedio de solución: {dt1_inversion} s"+"\n")
    d.write(f"Solución: {x} s"+"\n")
    
    #print ("Matriz de ", N)
    #print(f"Tiempo solución: {dt1_inversion} s")

d.close()

plt.loglog(Ks,dt1, marker = "o")
plt.ylabel("Tiempo Transcurrido (s)")
plt.xlabel("Tamaño Matiz N")
plt.loglog(Ns, dt,  marker = "o")

plt.legend(('float', 'double'),
           loc='upper right')

#Yticks (Tiempo Transcurrido (s))
labelsY1 = ["0,1 ms", "1 ms", "10 ms", "0,1 s", "1 s", "10 s", "1 min", "10 min"]
labels_Y1 = [10**(-4), 10**(-3), 10**(-2), 10**(-1), 10**(0), 10**(1), 10**(2), 10**(3) ]
plt.yticks(labels_Y1, labelsY1)

#Xticks
labelsX = ["10", "20", "50", "100", "200", "500", "1000", "2000", "5000", "10000", "20000" ]
xlabel2 = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
plt.xticks(xlabel2, labelsX, rotation = 45)

plt.grid(True)

plt.savefig("grafico_B_V_True.png")
plt.show()

-
-
-


























