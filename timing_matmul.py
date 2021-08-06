
from numpy import zeros
from time import perf_counter


n = 100
a = zeros((n, n))+1
b = zeros((n, n))+2

# print (f"a = {a}")
# print (f"a = {b}")

t1 = perf_counter()
c = a@b
t2 = perf_counter()

dt = t2 - t1
print (dt)
print (f"dt = (")
print (dt)
print (")s")













CODIGO P0E2
-----------------------------------------------------------------------------------------
FLOPS
def matmul (A,B):
    
    C = zeros((N, N))
    
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i,j] += a[i, k]*b[k, j] 
                
  


-----------------------------------------------------------------------------------------

RENDIMIENTO 

from numpy import zeros, float16, float32, float64
from time import perf_counter
import matplotlib.pylab as plt

N = 10000

Ns = [10, 100, 1000, 2000]

dts = []
mems = []

fid = open("rendimiento.txt", "w")


for N in Ns:
    
    # uso_memoria = 0
    
    
    
    A = zeros((N, N), dtype=float16 ) + 1
    
    
    
    
    uso_memoria_teorico = 2*N*N    #bytes...float32 4bytes
    uso_memoria_numpy = A.nbytes
    print(f"uso_memoria_teorico = {uso_memoria_teorico}")
    print(f"uso_memoria_numpy = {uso_memoria_numpy}")
    #exit(0)
    
    
    
    B = zeros((N, N)) + 2
    
    
    
    
    
    t1 = perf_counter()
    C = A@B
    t2 = perf_counter()
    
    uso_memoria_total = A.nbytes + B.nbytes + C.nbytes

    
    
    
    
    dt = t2 - t1
    
    dts.append(dt)
    mems.append(uso_memoria_total)
    
    print (f"N = {N} dt = {dt} s men = {uso_memoria_total} bytes flops = {N**3/dt} flops/s")
    
    fid.write(f"{N} {dt} {uso_memoria_total}\n")
    

fid.close()

plt.figure(1)
plt.subplot(2, 1, 1)
plt.loglog(Ns, dts)
plt.subplot(2, 1, 2)
plt.loglog(Ns, mems)
plt.show()


-----------------------------------------------------------------------------------------------
CODIGO GRAFICO

import matplotlib.pylab as plt

Ns = []
Ns1 = []
Ns2 = []
Ns3 = []
Ns4 = []
Ns5 = []
Ns6 = []
Ns7 = []
Ns8 = []
Ns9 = []


dts = []
dts1 = []
dts2 = []
dts3 = []
dts4 = []
dts5 = []
dts6 = []
dts7 = []
dts8 = []
dts9 = []

mems = []
mems1 = []
mems2 = []
mems3 = []
mems4 = []
mems5 = []
mems6 = []
mems7 = []
mems8 = []
mems9 = []





fid = open("rendimiento.txt","r")


for i in range (10):
    fid = open("rendimiento.txt","r")

    for line in fid:
        sl = line.split()
        N = int(sl[0])
        dt = float(sl[1])
        mem = int(sl[2])
        
        Ns.append(N)
        Ns1.append(N)
        Ns2.append(N)
        Ns3.append(N)
        Ns4.append(N)
        Ns5.append(N)
        Ns6.append(N)
        Ns7.append(N)
        Ns8.append(N)
        Ns9.append(N)
        
        dts.append(dt)
        dts1.append(dt)
        dts2.append(dt)
        dts3.append(dt)
        dts4.append(dt)
        dts5.append(dt)
        dts6.append(dt)
        dts7.append(dt)
        dts8.append(dt)
        dts9.append(dt)
        
        mems.append(mem)
        mems1.append(mem)
        mems2.append(mem)
        mems3.append(mem)
        mems4.append(mem)
        mems5.append(mem)
        mems6.append(mem)
        mems7.append(mem)
        mems8.append(mem)
        mems9.append(mem)
        
        #print(sl)
        
        
    fid.close()

# exit(0)



#Xticks
labelsX = ["10", "20", "50", "100", "200", "500", "1000", "2000", "5000", "10000" ]
xlabel2 = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]

#Yticks (Tiempo Transcurrido (s))
labelsY1 = ["0,1 ms", "1 ms", "10 ms", "0,1 s", "1 s", "10 s", "1 min", "10 min"]
labels_Y1 = [10**(-4), 10**(-3), 10**(-2), 10**(-1), 10**(0), 10**(1), 10**(2), 10**(3) ]

#Yticks (Uso Memoria (s))
labelsY2 = ["1 KB", "10 KB", "100 KB", "1 MB", "10 MB", "100 MB", "1 GB", "10 GB"]
mems_2 = [ 10**(3), 10**(4), 10**(5), 10**(6), 10**(7), 10**(8), 10**(9), 10**(10), 10**(11)] 




plt.figure(1)

plt.subplot(2, 1, 1)
plt.ylabel("Tiempo Transcurrido (s)")
plt.title("Rendimiento A@B")
plt.loglog(Ns, dts, marker = "o")
plt.loglog(Ns1, dts1, marker = "o")
plt.loglog(Ns2, dts2, marker = "o")
plt.loglog(Ns3, dts3, marker = "o")
plt.loglog(Ns4, dts4, marker = "o")
plt.loglog(Ns5, dts5, marker = "o")
plt.loglog(Ns6, dts6, marker = "o")
plt.loglog(Ns7, dts7, marker = "o")
plt.loglog(Ns8, dts8, marker = "o")
plt.loglog(Ns9, dts9, marker = "o")


plt.xticks(xlabel2, labelsX, rotation = 45)
plt.yticks(labels_Y1, labelsY1)
plt.grid(True)


plt.subplot(2, 1, 2)
plt.ylabel("Uso Memoria (s)")
plt.xlabel("Tamaño Matriz N")
plt.loglog(Ns, mems, marker = "o")
plt.loglog(Ns1, mems1, marker = "o")
plt.loglog(Ns2, mems2, marker = "o")
plt.loglog(Ns3, mems3, marker = "o")
plt.loglog(Ns4, mems4, marker = "o")
plt.loglog(Ns5, mems5, marker = "o")
plt.loglog(Ns6, mems6, marker = "o")
plt.loglog(Ns7, mems7, marker = "o")
plt.loglog(Ns8, mems8, marker = "o")
plt.loglog(Ns9, mems9, marker = "o")
plt.axhline(y = 8000000000, linestyle = "--", color = "black")

plt.xticks(xlabel2, labelsX, rotation = 45)
plt.yticks(mems_2, labelsY2)
plt.grid(True)



plt.show()
--------------------------------------------------------------------------------------------------





