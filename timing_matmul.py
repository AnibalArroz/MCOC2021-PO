
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








# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 11:05:24 2021

@author: Nicolas
"""

import matplotlib.pylab as plt

Ns = []
dts = []
mems = []

fid = open("rendimiento.txt","r")




for line in fid:
    sl = line.split()
    N = int(sl[0])
    dt = float(sl[1])
    mem = int(sl[2])
    
    Ns.append(N)
    dts.append(dt)
    mems.append(mem)
    
    #print(sl)


fid.close()

# exit(0)



#Xticks
labelsX = ["10", "20", "50", "100", "200", "500", "1000", "2000", "5000", "10000" ]
xlabel2 = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]

#Yticks (Tiempo Transcurrido (s))
labelsY1 = ["0,1 ms", "1 ms", "10 ms", "0,1 s", "1 s", "10 s", "1 min", "10 min"]
labels_Y1 = [10**(-4), 10**(-3), 10**(-2), 10**(-1), 10**(0), 10**(1), 10**(2) ]

#Yticks (Uso Memoria (s))
labelsY2 = ["1 KB", "10 KB", "100 KB", "1 MB", "10 MB", "100 MB", "1 GB", "10 GB"]
mems2 = [ 10**(3), 10**(4), 10**(5), 10**(6), 10**(7), 10**(8), 10**(9), 10**(10), 10**(11)] 




plt.figure(1)

plt.subplot(2, 1, 1)
plt.ylabel("Tiempo Transcurrido (s)")
plt.title("Rendimiento A@B")
plt.loglog(Ns, dts, marker = "o")
plt.xticks(xlabel2, labelsX, rotation = 45)
plt.yticks(labels_Y1, labelsY1)
plt.grid(True)


plt.subplot(2, 1, 2)
plt.ylabel("Uso Memoria (s)")
plt.xlabel("Tamaño Matriz N")
plt.loglog(Ns, mems, marker = "o")
plt.xticks(xlabel2, labelsX, rotation = 45)
plt.yticks(mems2, labelsY2)
plt.grid(True)



plt.show()

