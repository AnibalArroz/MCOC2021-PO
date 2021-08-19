# MCOC2021-PO
Mi Computador Principal



Marca/modelo: HP 340 G2

Tipo: Notebook

Año adquisición: 2016


Procesador:

-Marca/Modelo: Intel Core i5-5200U

-Velocidad Base: 2.20 GHz

-Velocidad Máxima: 2.20 GHz

-Numero de núcleos: 2

-Numero de hilos: 4

-Arquitectura: x86_64

-Set de instrucciones: Intel SSE4.1, Intel SSE4.2


Tamaño de las cachés del procesador 

-L1d: 32KB

-L1i: 32KB

-L2: 256KB

-L3: 3000KB


Memoria

-Total: 8 GB

-Tipo memoria: DDR3

-Velocidad 1600 MHz

-Numero de (SO)DIMM: 4


Tarjeta Gráfica

-Marca / Modelo: Intel(R) HD Graphics Family

-Memoria dedicada: 4 GB

-Resolución: 1366 x 768


Disco 1:

-Marca: Samsung

-Tipo: HDD

-Tamaño: 1TB

-Particiones: 1 (5 contando las de recuperación)

-Sistema de archivos: NTFS



Dirección MAC de la tarjeta wifi: 18-4F-32-CD-F2-81

Dirección IP (Interna, del router): 192.168.1.55

Dirección IP (Externa, del ISP): 190.215.234.150

Proveedor internet: GTD Manquehue. Fibra optica

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
-
-

P0E2
Preguntas

1)¿Cómo difiere del gráfico del profesor/ayudante?
- De algunos colores y del tipo de curva en el grafico de rendimiento.

2)¿A qué se pueden deber las diferencias en cada corrida?
- Puede deberse por las distintas y variadas tareas externas que esta haciendo el computador. 

3)El gráfico de uso de memoria es lineal con el tamaño de matriz, pero el de tiempo transcurrido no lo es ¿porqué puede ser?
- Es debido a que primero los procesadores usan las memorias "cache", luego los calculos mas complejos los toma otro segmento que trabaja mas rapido y asi. Esto es debido a que el computador asigna jerarquicamente los problemas y recursos viendo el contexto y las dimenciones del problema. 

4)¿Qué versión de python está usando?
- 3.7

5)¿Qué versión de numpy está usando?
- 2019.0.117

6)Durante la ejecución de su código ¿se utiliza más de un procesador? Muestre una imagen (screenshot) de su uso de procesador durante alguna corrida para confirmar. 

![image](https://user-images.githubusercontent.com/88512479/128574493-4e8d9fc7-5a64-4627-a67f-05d6a5ca87e1.png)

En mi caso se utiliza solamente un procesador

Desempeño MATMUL


![image](https://user-images.githubusercontent.com/88512479/128570964-fab84a73-c608-4070-b98e-4da0c2ad284e.png)




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
-
-
--
-
-
-
-

P0E3

codigo caso 2 timing_inv_caso_2_half



![image](https://user-images.githubusercontent.com/88512479/130005772-8134ea7c-95ce-4adc-b337-89602b51a812.png)


![image](https://user-images.githubusercontent.com/88512479/130005789-244bcc0c-ad9e-42db-9358-4403584e5a0a.png)

-
-
-

codigo caso 2 timing_inv_caso_2_single




![image](https://user-images.githubusercontent.com/88512479/130006085-eb5dde5a-fe28-45c7-a7ad-c18f1646cd8e.png)


![image](https://user-images.githubusercontent.com/88512479/130006092-359e8e45-d839-467f-8a93-f61797b4afff.png)


-
-
-

codigo caso 2 timing_inv_caso_2_double



![image](https://user-images.githubusercontent.com/88512479/130006442-4dc29569-d353-4fbe-b1e2-05a204e051d9.png)

![image](https://user-images.githubusercontent.com/88512479/130006450-06540ed7-d68a-44b7-93c8-a59b689594b2.png)


-
-
-

codigo caso 2 timing_inv_caso_2_longdouble




![image](https://user-images.githubusercontent.com/88512479/130006675-dff86449-6e3c-4a70-acd0-77c2a3489355.png)

![image](https://user-images.githubusercontent.com/88512479/130006848-25c105ad-8632-460c-a10b-2ed657fb3232.png)


-
-
-

codigo caso 3 timing_inv_caso_3_half

![image](https://user-images.githubusercontent.com/88512479/130007925-9a46fa2b-50b4-42ee-8428-55826ce70c9d.png)


![image](https://user-images.githubusercontent.com/88512479/130007937-d927479f-50c2-4f6a-9098-570374454bc6.png)


-
-
-

codigo caso 3 timing_inv_caso_3_single

![image](https://user-images.githubusercontent.com/88512479/130008097-da3656da-66fe-4a60-9d72-edf52791b6d0.png)


![image](https://user-images.githubusercontent.com/88512479/130008106-29a201e6-2d27-4597-8d09-99d1a03306bd.png)

-
-
-


codigo caso 3 timing_inv_caso_3_double

![image](https://user-images.githubusercontent.com/88512479/130008548-e9826a71-e532-4074-b3b2-80d0a6fb0cde.png)


![image](https://user-images.githubusercontent.com/88512479/130008559-f22a0b9f-9c92-4e5a-9780-2e7713d73d26.png)


-
-
-

codigo caso 3 timing_inv_caso_3_longdouble

![image](https://user-images.githubusercontent.com/88512479/130008865-27e23b12-a337-4866-a202-15efae9eed56.png)


![image](https://user-images.githubusercontent.com/88512479/130008879-50854716-2e0a-4552-96b2-9ecc54082d61.png)


-
-
-

codigo caso 1 timing_inv_caso_1_half

ERROR

-
-
-

codigo caso 1 timing_inv_caso_1_single

![image](https://user-images.githubusercontent.com/88512479/130009712-7327f4c7-34e3-4e87-a2c0-a53133aeefa4.png)


![image](https://user-images.githubusercontent.com/88512479/130009728-6dee348d-5eb0-4024-a7cc-155ff9dac102.png)

-
-
-

codigo caso 1 timing_inv_caso_1_double


![image](https://user-images.githubusercontent.com/88512479/130010022-d89cc496-f685-4fb8-8895-bcf93c428559.png)


![image](https://user-images.githubusercontent.com/88512479/130010032-3b2a12a9-d381-47ea-bc35-f834065fd1e6.png)

-
-
-

codigo caso 1 timing_inv_caso_1_longdouble

ERROR




PREGUNTAS A RESPONDER







