

1)- Comente las diferencias que ve en el comportamiento de los algoritmos en el caso de matrices llenas y dispersas.

- En cuanto a las matrices llenas, estas se vieron un poco mas lentas que las dispersas debido a su gran numero de ceros(0) que tuvieron que almacenar. 
- Las dispersas eran mas veloces de resolver ya que tenian menos datos que leer.
- Por otro lado, hablando acerca del solve y de la inversion; Se vio que el ensamblado de "Solve", demoro mas, debido a que tuvo que ensamblar una matriz "A", y un vectos de puros unos "B". En cambio el metodo "inv" se nos pedia ensamblar solamente una matriz "A". 
- En cuanto a la solucion, esta vez se noto mas lenta el metodo "inv" devido que el proceso de invertir una matriz es considerablemente mas trabajoso que solo multiplicarla "@". 


2)- ¿Cual parece la complejidad asintótica (para N→∞)  para el ensamblado y solución en ambos casos y porqué?

- Entre mas grande el N, mas compleja se va haciendo resolver la matriz debido a su tamaño y gran trabajo en el calculo. 
- Ante un N mas grande, el N^ tendera tambien a ir a un "x" mas grande para el caso de ensamblado y solución.


3)- ¿Como afecta el tamaño de las matrices al comportamiento aparente?

- Al ser mayor la matriz, mas complejo se hara el desarrollo y el calculo de lo pedido, por lo que el computador tardara mas en obtener el resultado. 


4)- ¿Qué tan estables son las corridas (se parecen todas entre si siempre, nunca, en un rango)?

- Como se ve en la siguiente imagen en la que estan los 4 graficos, los ensamblajes en el caso de las matrices que habia que obtener las inversas, son bastantes parecidos, y el caso de solve, tambien parecidos. 
- En cuanto al grafico de solucion, las matrices inversas tambien son parecidas entre si y las matrices solve tienen su propio parecido. 
- ![image](https://user-images.githubusercontent.com/88512479/131780527-d5b883f9-0b1b-4c2b-aab5-cec5383c51f1.png)




A continuación, formateare el codigo de mi matriz laplaciana. 

```

from numpy import zeros, float64
import numpy as np
import scipy.sparse as sparse

 
    
    
def matriz_laplaciana(N, t=float64):
    d = sparse.eye(N, N, 1,  dtype = t) - sparse.eye(N, N, 1, dtype = t)
    return 2*sparse.eye(N, dtype=t) -d - d.T




N =10


A = matriz_laplaciana(N)

```


La eleccion de la matriz laplaciana es de gran importancia debido que con ella se trabajara todo y segun su simplicidad de crear matrices, nos otorgara una mayor efectividad con respecto al tiempo de solución. 

















