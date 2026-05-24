import numpy as np
import matplotlib.pyplot as plt
import time

# inciso a)

#parametros

f1 = 100.0
f2 = 200.0
N = 1000
T = 0.05

t_n = np.linspace(0, T, N, endpoint=False)

x_n = np.sin(2 * np.pi * f1 * t_n) + 0.5 * np.sin(2 * np.pi * f2 * t_n)


#inciso b)

X_k = np.zeros(N ,dtype = complex)

for k in range(N):
    
    suma_act = 0.0 + 0.0j
    
    for n in range(N):
        
        exponente = -1j * 2 * np.pi * k * n / N
        
        suma_act += x_n[n] * np.exp(exponente)
        
    X_k[k] = suma_act
    
#inciso c)

espectro = np.abs(X_k)

frecuencias = np.arange(N) / T

plt.figure(figsize = (10, 5))
plt.plot(frecuencias, espectro)
plt.title("Espectro de Frecuencias (DFT manual)")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud |X_k|")
plt.xlim(0, 300)
plt.grid()
plt.show()
       

#Inciso d)

X_k_fft = np.fft.fft(x_n)

#Inciso e)

valor_N = [100, 1000, 10000]
iteraciones = 15

tiempos_dft_promedio = []

tiempos_fft_promedio = []

for i in valor_N:
    
    tiempos_dft_temp = []
    
    tiempos_fft_temp = []
    
    print(f"\nIniciando calculos para N = {i}")
    
    for j in range(iteraciones):
        
        t_n = np.linspace(0, T, i, endpoint=False)
        
        x_n = np.sin(2 * np.pi * f1 * t_n) + 0.5 * np.sin(2 * np.pi * f2 * t_n)
        
        inicio_dft = time.time()
        
        X_k = np.zeros(i, dtype=complex)
        
        for k in range(i):
            
            suma_act = 0.0 + 0.0j
            
            for n in range(i):
                
                exponente = -1j * 2 * np.pi * k * n / i
                
                suma_act += x_n[n] * np.exp(exponente)
                
            X_k[k] = suma_act
            
        fin_dft = time.time()
        
        tiempos_dft_temp.append(fin_dft - inicio_dft)
        
        inicio_fft = time.time()
        
        X_k_fft = np.fft.fft(x_n)
        
        fin_fft = time.time()
        
        tiempos_fft_temp.append(fin_fft - inicio_fft)
        
        print(f"  Iteracion {j+1}/{iteraciones} completada")
        
    promedio_dft = np.mean(tiempos_dft_temp)
    promedio_fft = np.mean(tiempos_fft_temp)
    
    tiempos_dft_promedio.append(promedio_dft)
    tiempos_fft_promedio.append(promedio_fft)
    
    print(f"N = {i}: DFT prom = {promedio_dft:.4f} s, FFT prom = {promedio_fft:.6f} s")

#Inciso f)

plt.figure(figsize=(8, 5))
plt.plot(valor_N, tiempos_dft_promedio, marker='o', label='DFT Manual')
plt.plot(valor_N, tiempos_fft_promedio, marker='s', label='FFT (NumPy)')
plt.title("Tiempo de Ejecucion vs N")
plt.xlabel("N")
plt.ylabel("Tiempo Promedio (s)")
plt.legend()
plt.grid(True)
plt.show()

# Inciso g)

plt.figure(figsize=(8, 5))
plt.loglog(valor_N, tiempos_dft_promedio, marker='o', label='DFT Manual')
plt.loglog(valor_N, tiempos_fft_promedio, marker='s', label='FFT (NumPy)')
plt.title("Tiempo de Ejecucion vs N (Escala Log-Log)")
plt.xlabel("N")
plt.ylabel("Tiempo Promedio (s)")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()

log_N1 = np.log10(valor_N[-2])
log_N2 = np.log10(valor_N[-1])

log_t1_dft = np.log10(tiempos_dft_promedio[-2])
log_t2_dft = np.log10(tiempos_dft_promedio[-1])
exponente_dft = (log_t2_dft - log_t1_dft) / (log_N2 - log_N1)

log_t1_fft = np.log10(tiempos_fft_promedio[-2] + 1e-10)
log_t2_fft = np.log10(tiempos_fft_promedio[-1] + 1e-10)
exponente_fft = (log_t2_fft - log_t1_fft) / (log_N2 - log_N1)

print(f"Exponente de escalamiento experimental DFT: {exponente_dft:.2f}")
print(f"Exponente de escalamiento experimental FFT: {exponente_fft:.2f}")

# Inciso h)

"""
Al evaluar la razón empírica de los tiempos de ejecución t(DFT)/t(FFT)
extraídos de la simulación, se observa que el algoritmo FFT se vuelve al menos
100 veces más rápido en el entorno de $N \approx 10^3$ (el valor exacto 
dependerá de las constantes del hardware utilizado). A partir de esta escala, 
la complejidad cuadrática O(N^2) de la DFT directa vuelve intratable el cómputo
en comparación con el escalamiento casi lineal de O(N \log N) de la FFT.

"""

# Inciso i)

"""

En simulaciones de dinámica cuántica (como propagar paquetes de onda para la 
ecuación de Schrödinger dependiente del tiempo), es común usar el método del 
Operador de División (Split-Operator). Este método requiere transformar la 
función de onda desde el espacio de posiciones al espacio de momentum 
(y viceversa) en cada pequeño paso temporal. Sin la FFT, el costo computacional
de realizar estas transformaciones anularía por completo la ventaja del método,
impidiendo simular sistemas incluso de unas pocas partículas.

"""










































