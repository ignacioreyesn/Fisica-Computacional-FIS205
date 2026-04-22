import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

kb = 1.380649e-23

m = 3.32e-27

N = 125

L_actual = 10e-6

T0 = 300

dt = 1e-12

epsilon =33.3 * kb

sigma = 0.296e-9

def posiciones_iniciales(N, L_actual):
    
    n = int(np.round(N**(1/3)))
    
    espaciado = L_actual / n
    
    posiciones = []
    
    for i in range(n):
        for j in range(n):
            for k in range(n):
                
                x = (i + 0.5) * espaciado + np.random.uniform(-0.1, 0.1) * espaciado
                y = (j + 0.5) * espaciado + np.random.uniform(-0.1, 0.1) * espaciado
                z = (k + 0.5) * espaciado + np.random.uniform(-0.1, 0.1) * espaciado
                
                posiciones.append([x, y ,z])
                
    return np.array(posiciones)

posiciones = posiciones_iniciales(N, L_actual)

v = np.random.normal(0, 1, (N, 3))

v_cm = np.mean(v, axis = 0)

v -= v_cm

energia_cinetica = 0.5 * m *np.sum(v**2)

T_actual =(2 / 3) * energia_cinetica / (N * kb)

factor_termico = np.sqrt(T0 / T_actual)

v *= factor_termico

pasos_totales = 1000
    
def calcular_fuerzas_energia(pos, eps, sig):
    
        dr = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        
        r2= np.sum(dr**2, axis=-1)
        
        np.fill_diagonal(r2, np.inf)
        
        sr2 = (sig**2 / 2)
        sr6 = sr2**3
        sr12 = sr6**2
        
        f_mag = 24 * eps * (2 * sr12 - sr6) / r2
        
        fuerzas = np.sum(f_mag[:, :, np.newaxis] * dr, axis = 1)
        
        energia_p = np.sum(4 * eps * (sr12 - sr6)) / 2
        
        return fuerzas, energia_p
    
historial_T = []
historial_Ec = []
historial_Ep = []
historial_presion = []
tiempo_arr = []
tiempo_actual = 0.0
ventana_datos = 500

fig, axs = plt.subplots(2, 2, figsize=(12, 8))
plt.subplots_adjust(bottom=0.25, hspace=0.3, wspace=0.3)

line_T, = axs[0, 0].plot([], [], color='red')
axs[0, 0].set_title('Temperatura vs Tiempo')
axs[0, 0].set_ylabel('T (K)')
axs[0, 0].grid(True, alpha=0.3)

line_P, = axs[0, 1].plot([], [], color='blue')
axs[0, 1].set_title('Presión vs Tiempo')
axs[0, 1].set_ylabel('P (Pa)')
axs[0, 1].grid(True, alpha=0.3)

line_Ec, = axs[1, 0].plot([], [], color='orange', label='E. Cinética')
line_Ep, = axs[1, 0].plot([], [], color='green', label='E. Potencial')
line_Et, = axs[1, 0].plot([], [], color='black', linestyle='--', label='E. Total')
axs[1, 0].set_title('Energías vs Tiempo')
axs[1, 0].set_ylabel('Energía (J)')
axs[1, 0].legend(loc='upper right', fontsize=8)
axs[1, 0].grid(True, alpha=0.3)

ax_hist = axs[1, 1]

ax_temp = plt.axes([0.15, 0.1, 0.65, 0.03])
slider_temp = Slider(ax_temp, 'T (K)', 10.0, 1000.0, valinit=T0)

ax_vol = plt.axes([0.15, 0.05, 0.65, 0.03])
slider_vol = Slider(ax_vol, 'Lado (m)', 5e-6, 20e-6, valinit=L_actual)

def update_temp(val):
    global v
    T_objetivo = slider_temp.val
    E_c_inst = 0.5 * m * np.sum(v**2)
    T_inst = (2.0 / 3.0) * E_c_inst / (N * kb)
    if T_inst > 0:
        v *= np.sqrt(T_objetivo / T_inst)

slider_temp.on_changed(update_temp)

def actualizar_frame(frame):
    global posiciones, v, tiempo_actual, L_actual
    
    L_actual = slider_vol.val
    area_total = 6 * (L_actual**2)
    
    pasos_por_frame = 10 
    
    for _ in range(pasos_por_frame):
        fuerzas, E_p = calcular_fuerzas_energia(posiciones, epsilon, sigma)
        
        v += (fuerzas / m) * dt
        posiciones += v * dt
        
        fuera_inf = posiciones < 0
        fuera_sup = posiciones > L_actual
        
        dp_col = np.sum(2 * m * np.abs(v[fuera_inf])) + np.sum(2 * m * np.abs(v[fuera_sup]))
        presion_inst = (dp_col / dt) / area_total
        
        v[fuera_inf] *= -1
        v[fuera_sup] *= -1
        
        posiciones[fuera_inf] = -posiciones[fuera_inf]
        posiciones[fuera_sup] = 2 * L_actual - posiciones[fuera_sup]
        
        E_c = 0.5 * m * np.sum(v**2)
        T_inst = (2.0 / 3.0) * E_c / (N * kb)
        tiempo_actual += dt
        
    historial_T.append(T_inst)
    historial_Ec.append(E_c)
    historial_Ep.append(E_p)
    historial_presion.append(presion_inst)
    tiempo_arr.append(tiempo_actual)
    
    if len(historial_T) > ventana_datos:
        historial_T.pop(0)
        historial_Ec.pop(0)
        historial_Ep.pop(0)
        historial_presion.pop(0)
        tiempo_arr.pop(0)
        
    t_min, t_max = tiempo_arr[0], tiempo_arr[-1]
    
    line_T.set_data(tiempo_arr, historial_T)
    axs[0, 0].set_xlim(t_min, t_max)
    axs[0, 0].set_ylim(min(historial_T)*0.9, max(historial_T)*1.1)
    
    line_P.set_data(tiempo_arr, historial_presion)
    axs[0, 1].set_xlim(t_min, t_max)
    axs[0, 1].set_ylim(0, max(max(historial_presion)*1.1, 1e-6))
    
    line_Ec.set_data(tiempo_arr, historial_Ec)
    line_Ep.set_data(tiempo_arr, historial_Ep)
    E_tot_arr = np.array(historial_Ec) + np.array(historial_Ep)
    line_Et.set_data(tiempo_arr, E_tot_arr)
    axs[1, 0].set_xlim(t_min, t_max)
    axs[1, 0].set_ylim(min(0, min(historial_Ep)*1.1), max(E_tot_arr)*1.2)
    
    if frame % 5 == 0:
        ax_hist.clear()
        v_mags = np.linalg.norm(v, axis=1)
        ax_hist.hist(v_mags, bins=15, density=True, color='purple', alpha=0.6, label='Simulación')
        
        v_teo = np.linspace(0, np.max(v_mags)*1.5, 100)
        term1 = 4 * np.pi * v_teo**2
        term2 = (m / (2 * np.pi * kb * T_inst))**(1.5)
        term3 = np.exp(-m * v_teo**2 / (2 * kb * T_inst))
        ax_hist.plot(v_teo, term1 * term2 * term3, color='black', label='Maxwell-Boltzmann')
        
        ax_hist.set_title('Distribución de Velocidades')
        ax_hist.set_xlabel('v (m/s)')
        ax_hist.legend(fontsize=8)
        
    return line_T, line_P, line_Ec, line_Ep, line_Et

ani = FuncAnimation(fig, actualizar_frame, interval=50, blit=False, cache_frame_data=False)
plt.show()

