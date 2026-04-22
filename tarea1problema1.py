#Problema 1 - Modelo de Ising#

"""
a) Para un N dado, la dimensión del espacio de Hilbert del sistema completo es
    de 2^N , mientras que el espacio de operadores, cuya representación es de
    matrices cuadradas, es de 2^N * 2^N.
"""

"""
b) Construcción de Hamiltoniano de Ising
"""

import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
import time
from scipy.optimize import curve_fit

#base de estados vector

up = np.array([[1],
               [0]])

down = np.array([[0],
                 [1]])

#base de operadores

I = np.array([[1, 0],
              [0, 1]])

sigmax = np.array([[0, 1],
                   [1, 0]])

sigmay = np.array([[0, -1j],
                    [1j, 0]])

sigmaz = np.array([[1, 0],
                   [0, -1]])

#parametros temporales

t_i = 0.0
t_f = 15.0
dt = 0.05

tiempos = np.arange(t_i, t_f, dt)

#parametros sistema

N = 5
J = 1.0

def transversal(i, N):
    
    if i == 0:
        termino = sigmaz
    else:
        termino = I
    for j in range(1, N):
        
        if j == i:
            termino = np.kron(termino, sigmaz)
        else:
            termino = np.kron(termino, I)
            
    return termino

def interaccion(i, N):
    
    if i == 0:
        termino2 = sigmax
    else:
        termino2 = I
    for j in range(1, N):
        
        if j == i or j ==i +1:
            termino2 = np.kron(termino2, sigmax)
        else:
            termino2 = np.kron(termino2, I)
    
    return termino2

def hamiltoniano(B, J, N):
    
    dimension = 2**N
    
    H = np.zeros((dimension, dimension), dtype=complex)
    
    for i in range(N):
        H = H + B * transversal(i, N)
    for i in range(N-1):
        H = H + J * interaccion(i, N)
        
    return H

"""
c) Evolución temporal y probabilidad
"""

def downinicial(N):
    estado = down
    for k in range(1, N):
        estado = np.kron(estado, down)
    return estado


casos_B =  [0.1, 1.0, 10.0]

for B in casos_B:
    
    H = hamiltoniano(B, J, N)
    U = expm(-1j * H * dt) 
    
    estado_0 = downinicial(N)
    estado_actual = downinicial(N)
    
    probabilidades = []
    
    for t in tiempos:
        
        amplitud = np.vdot(estado_0, estado_actual)
        probabilidades.append(np.abs(amplitud)**2)
        
        estado_actual = U @ estado_actual
        
    plt.plot(tiempos, probabilidades, label=rf'$B/J = {B/J}$')
    
    
plt.title(f'Probabilidad de retorno al estado inicial para N={N} espines')
plt.xlabel('Tiempo ($t$)')
plt.ylabel(r'Probabilidad $p(t) = |\langle\Psi(t)|\Psi(0)\rangle|^2$')
plt.legend(loc='upper right', shadow=True)
plt.grid(True, linestyle='--', alpha=0.7)
plt.ylim(0, 1.05)

plt.show()
    
    
"""
d) tiempo de cálculo en diagonalizar - e) Tiempo de ejecución t/N
"""

valores_N = [4, 5, 6, 7, 8]
tiempos_promedio = []
numero_realizaciones = 5 


J = 1.0
B = 1.0

for N_prueba in valores_N:
    
    tiempo_acumulado = 0.0
    
    for _ in range(numero_realizaciones):
        
        inicio = time.time()
        
        
        H = hamiltoniano(B, J, N_prueba)
        autovalores, autovectores = np.linalg.eigh(H)
        
        fin = time.time()
        
        tiempo_acumulado = tiempo_acumulado + (fin - inicio)
        
    tiempo_medio = tiempo_acumulado / numero_realizaciones
    tiempos_promedio.append(tiempo_medio)
    
    print(f"N = {N_prueba} | Dimensión: {2**N_prueba}x{2**N_prueba} | Tiempo Medio: {tiempo_medio:.5f} s")

plt.figure(figsize=(8, 5))

plt.plot(valores_N, tiempos_promedio, marker='o', color='red', label='Tiempo de ejecución')

plt.title('Costo computacional de diagonalización exacta')
plt.xlabel('Número de espines ($N$)')
plt.ylabel('Tiempo promedio de ejecución (s)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.show()


"""
f) estimación para N grande - g) comparación con edad del universo
"""

def modelo_exponencial(N, a, b):
    return a * np.exp(b * N)

parametros_optimos, covarianza = curve_fit(modelo_exponencial, valores_N, tiempos_promedio)

a_opt = parametros_optimos[0]
b_opt = parametros_optimos[1]

print("\n--- Resultados del Ajuste ---")
print(f"Modelo ajustado: T(N) = {a_opt:.2e} * exp({b_opt:.4f} * N)")

N_extrapolar = [20, 50, 100]
tiempos_estimados = []

for N_futuro in N_extrapolar:
    t_est = modelo_exponencial(N_futuro, a_opt, b_opt)
    tiempos_estimados.append(t_est)
    
    print(f"Estimación para N={N_futuro}: {t_est:.2e} segundos")

edad_universo_s = 4.3e17

print("\n--- Comparación con la Edad del Universo ---")
for i, N_futuro in enumerate(N_extrapolar):
    razon = tiempos_estimados[i] / edad_universo_s
    print(f"Para N={N_futuro}, la simulación tomaría {razon:.2e} veces la edad del Universo.")


"""
h) Desafíos del crecimiento exponencial

    
El crecimiento exponencial es una gran barrera para la simulación de sistemas
cuánticos de muchos cuerpos. Esto se debe a que, en este problema, estamos
trabajando con el sistema cuántico más simple posible: un sistema de dos
niveles. En este ejemplo de juguete ya se puede apreciar que, al acoplar muchos
de estos sistemas, el costo computacional se vuelve rápidamente inviable.

Ahora bien, para un caso más realista, se deben considerar muchos otros 
fenómenos y grados de libertad. Por ejemplo, para describir un único electrón
de forma completa, se debe realizar el producto tensorial de los espacios
asignados a sus cuatro números cuánticos (n, l, m y s), y esto sin siquiera 
proyectar sobre bases continuas (no numerables), como lo son las coordenadas
espaciales del sistema. Si quisiéramos obtener una descripción exacta de un
átomo multielectrónico pesado, como el uranio, resulta computacionalmente 
imposible tomar en cuenta cada una de las variables. 

El panorama es aún peor si queremos considerar interacciones sobre un gas 
macroscópico, en donde el número de átomos es del orden del número de Avogadro.
Es por esto que, para describir este tipo de sistemas, la física clásica y 
computacional estándar no bastan; se hace estrictamente necesario recurrir a 
métodos aproximados (como la física estadística de campos o métodos 
variacionales) o apostar por el desarrollo de simuladores y computadores 
cuánticos que operen nativamente en estos espacios.
    
"""    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
            
        
            