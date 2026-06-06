import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import sys

plt.close('all')

def definir_campos(t):
    Omega_0 = 5.0  
    Omega_1 = 0.5  
    omega = 5.0 
    
    Omega_x = Omega_1 * np.cos(omega * t)
    Omega_y = Omega_1 * np.sin(omega * t) 
    Omega_z = Omega_0
    
    return np.array([Omega_x, Omega_y, Omega_z])

def construir_lindblad(g_em, g_ab, g_de):
    Gamma = np.array([
        [-(g_em + g_ab) / 2 - 2 * g_de, 0, 0],
        [0, -(g_em + g_ab) / 2 - 2 * g_de, 0],
        [0, 0, -(g_em + g_ab)]
    ])
    
    c = np.array([0, 0, -(g_em - g_ab)])
    
    return Gamma, c

def derivada_bloch_lindblad(t, r, Gamma_mat, c_vec):
    Omega_vector = definir_campos(t)
    Ox, Oy, Oz = Omega_vector
    
    M = np.array([
        [0, -Oz, Oy],
        [Oz, 0, -Ox],
        [-Oy, Ox, 0]
    ])
    
    return (M + Gamma_mat) @ r + c_vec

def integrador_rk4(func, t_eval, r0, Gamma_mat, c_vec):
    n_steps = len(t_eval)
    r_out = np.zeros((3, n_steps))
    r_out[:, 0] = r0
    
    for i in range(n_steps - 1):
        t = t_eval[i]
        dt = t_eval[i+1] - t
        r = r_out[:, i]
        
        k1 = func(t, r, Gamma_mat, c_vec)
        k2 = func(t + dt/2, r + dt*k1/2, Gamma_mat, c_vec)
        k3 = func(t + dt/2, r + dt*k2/2, Gamma_mat, c_vec)
        k4 = func(t + dt, r + dt*k3, Gamma_mat, c_vec)
        
        r_out[:, i+1] = r + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
        
    return r_out

t_max = 15.0
fps = 30
phi = np.pi / 4
theta = np.pi / 6

gamma_emision = 0.15
gamma_absorcion = 0.05
gamma_dephasing = 0.10

Gamma_mat, c_vec = construir_lindblad(gamma_emision, gamma_absorcion, gamma_dephasing)

estado_inicial = np.array([np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)])
frames_total = int(t_max * fps)
t_eval = np.linspace(0, t_max, frames_total)

r_data = integrador_rk4(derivada_bloch_lindblad, t_eval, estado_inicial, Gamma_mat, c_vec)

t1_inv = gamma_emision + gamma_absorcion
t2_inv = (gamma_emision + gamma_absorcion) / 2 + 2 * gamma_dephasing

T1 = 1 / t1_inv if t1_inv > 0 else np.inf
T2 = 1 / t2_inv if t2_inv > 0 else np.inf

normas = np.linalg.norm(r_data, axis=0)
max_norm = np.max(normas)
min_norm = np.min(normas)

reporte = []
reporte.append("="*50)
reporte.append("REPORTE DE SIMULACION Y VALIDACION FISICA")
reporte.append("="*50)
reporte.append("PARAMETROS FENOMENOLOGICOS:")
reporte.append(f"T1 (Relajacion longitudinal) : {T1:.4f} s")
reporte.append(f"T2 (Decaimiento transversal) : {T2:.4f} s")
reporte.append("-" * 50)
reporte.append("VALIDACION GEOMETRICA Y PROBABILISTICA:")
reporte.append(f"Norma maxima alcanzada: {max_norm:.6f}")
reporte.append(f"Norma minima alcanzada: {min_norm:.6f}")

if max_norm <= 1.000001:
    reporte.append("CHECK SUPERADO: La desigualdad de Lindblad se cumple (|r| <= 1).")
else:
    reporte.append("ERROR: El vector sale de la esfera de Bloch (|r| > 1).")

if (gamma_emision + gamma_absorcion + gamma_dephasing) == 0:
    if np.allclose(normas, 1.0, atol=1e-5):
        reporte.append("CHECK SUPERADO: El sistema cerrado conserva la pureza estricta (|r| = 1).")
    else:
        reporte.append("ERROR: La pureza no se conserva.")
reporte.append("="*50)

texto_reporte = "\n".join(reporte)
print(texto_reporte, flush=True)

with open("reporte_simulacion.txt", "w") as f:
    f.write(texto_reporte)

fig_2d, axs = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
componentes = [r'$r_x(t)$', r'$r_y(t)$', r'$r_z(t)$']
colores = ['crimson', 'forestgreen', 'royalblue']

for i in range(3):
    axs[i].plot(t_eval, r_data[i, :], color=colores[i], linewidth=1.5)
    axs[i].set_ylabel(componentes[i], fontsize=12)
    axs[i].grid(True, alpha=0.3)
axs[2].set_xlabel('Tiempo (s)', fontsize=12)
fig_2d.suptitle('Dinamica Abierta del Vector de Bloch (RK4)', fontsize=14)
plt.tight_layout()

def get_arc_points_solid(r_vec, n_points=30, radius=0.5):
    x, y, z = r_vec
    z_clamped = np.clip(z, -1, 1)
    
    phi_curr = np.arctan2(y, x)
    phis = np.linspace(0, phi_curr, n_points)
    arc_phi_x = radius * np.cos(phis)
    arc_phi_y = radius * np.sin(phis)
    arc_phi_z = np.zeros_like(phis)
    verts_phi = [list(zip(np.append(arc_phi_x, 0), np.append(arc_phi_y, 0), np.append(arc_phi_z, 0)))]

    theta_curr = np.arccos(z_clamped)
    thetas = np.linspace(0, theta_curr, n_points)
    r_local = radius * np.sin(thetas)
    z_local = radius * np.cos(thetas)
    
    arc_theta_x = r_local * np.cos(phi_curr)
    arc_theta_y = r_local * np.sin(phi_curr)
    arc_theta_z = z_local
    verts_theta = [list(zip(np.append(arc_theta_x, 0), np.append(arc_theta_y, 0), np.append(arc_theta_z, 0)))]
    
    return verts_theta, verts_phi

def get_spinning_arrow_geometry(r_vec, phase_angle, scale=1.0):
    len_r = np.linalg.norm(r_vec)
    if len_r < 1e-6: return np.array([[0,0,0],[0,0,0]]), []
    u = r_vec / len_r
    not_u = np.array([1, 0, 0]) if np.abs(u[0]) < 0.9 else np.array([0, 1, 0])
    v = np.cross(u, not_u)
    v /= np.linalg.norm(v)
    w = np.cross(u, v)
    c, s = np.cos(phase_angle), np.sin(phase_angle)
    v_rot = c * v + s * w
    w_rot = -s * v + c * w
    shaft = np.array([[0, 0, 0], r_vec])
    head_size = 0.15 * scale
    base = r_vec - head_size * u
    fins = []
    f = 0.5 
    dirs = [v_rot, w_rot, -v_rot, -w_rot, v_rot]
    for i in range(4):
        p1 = base + (head_size * f) * dirs[i]
        p2 = base + (head_size * f) * dirs[i+1]
        fins.append(np.array([p1, p2]))
        fins.append(np.array([p1, r_vec]))
    return shaft, fins

fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(111, projection='3d')
ax.set_axis_off()

u_sphere = np.linspace(0, 2 * np.pi, 50)
v_sphere = np.linspace(0, np.pi, 25)
ax.plot_wireframe(np.outer(np.cos(u_sphere), np.sin(v_sphere)), 
                  np.outer(np.sin(u_sphere), np.sin(v_sphere)), 
                  np.outer(np.ones(np.size(u_sphere)), np.cos(v_sphere)), 
                  color="gray", alpha=0.15, linewidth=0.8)

L = 1.3
ax.plot([-L, L], [0, 0], [0, 0], 'k-', linewidth=1, alpha=0.8)
ax.plot([0, 0], [-L, L], [0, 0], 'k-', linewidth=1, alpha=0.8)
ax.plot([0, 0], [0, 0], [-L, L], 'k-', linewidth=1, alpha=0.8)

ax.text(L, 0, 0, r"$|+x\rangle$", fontsize=14, fontweight='bold')
ax.text(-L, 0, 0, r"$|-x\rangle$", fontsize=14, fontweight='bold')
ax.text(0, L, 0, r"$|+y\rangle$", fontsize=14, fontweight='bold')
ax.text(0, -L, 0, r"$|-y\rangle$", fontsize=14, fontweight='bold')
ax.text(0, 0, L, r"$|0\rangle$", fontsize=14, fontweight='bold')
ax.text(0, 0, -L, r"$|1\rangle$", fontsize=14, fontweight='bold')

trayectoria, = ax.plot([], [], [], color="blue", linewidth=1.5, linestyle=':')
shaft_line, = ax.plot([], [], [], color='crimson', linewidth=3)
fin_lines = [ax.plot([], [], [], color='crimson', linewidth=2)[0] for _ in range(8)]

theta_poly = Poly3DCollection([], alpha=0.4, facecolor='salmon', edgecolor='red')
phi_poly = Poly3DCollection([], alpha=0.4, facecolor='skyblue', edgecolor='blue')
ax.add_collection3d(theta_poly)
ax.add_collection3d(phi_poly)

title_text = ax.text2D(0.5, 0.95, "", transform=ax.transAxes, ha="center", fontsize=14)

matrix_text = ax.text2D(0.02, 0.88, "", transform=ax.transAxes, ha="left", va="top", 
                        fontsize=11, family='monospace', 
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="lightgray"))

ax.view_init(elev=15, azim=30)
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])

phase_acumulada = 0.0

def update(frame):
    global phase_acumulada
    
    curr_r = r_data[:, frame]
    
    trayectoria.set_data(r_data[0, :frame+1], r_data[1, :frame+1])
    trayectoria.set_3d_properties(r_data[2, :frame+1])
    
    phase_acumulada -= 0.5 
    shaft, fins = get_spinning_arrow_geometry(curr_r, phase_acumulada)
    shaft_line.set_data(shaft[:, 0], shaft[:, 1])
    shaft_line.set_3d_properties(shaft[:, 2])
    for i, line in enumerate(fin_lines):
        line.set_data(fins[i][:, 0], fins[i][:, 1]) if i < len(fins) else line.set_data([], [])
        line.set_3d_properties(fins[i][:, 2]) if i < len(fins) else line.set_3d_properties([])

    v_theta, v_phi = get_arc_points_solid(curr_r, radius=0.5)
    theta_poly.set_verts(v_theta)
    phi_poly.set_verts(v_phi)

    Omega_vector = definir_campos(t_eval[frame])
    intensidad = np.linalg.norm(Omega_vector)
    
    norm_r = np.linalg.norm(curr_r)
    z_clamped = np.clip(curr_r[2] / norm_r, -1, 1) if norm_r > 1e-9 else 0
    theta_deg = np.degrees(np.arccos(z_clamped))
    phi_deg = np.degrees(np.arctan2(curr_r[1], curr_r[0]))

    title_text.set_text(f"Tiempo: {t_eval[frame]:.2f}s | $|\Omega|$: {intensidad:.2f} | $|r|$: {norm_r:.2f}\n"
                        f"$\\theta$: {theta_deg:.0f}° | $\\phi$: {phi_deg:.0f}°")

    rho_00 = 0.5 * (1 + curr_r[2])
    rho_11 = 0.5 * (1 - curr_r[2])
    rho_01 = 0.5 * (curr_r[0] - 1j * curr_r[1])
    rho_10 = 0.5 * (curr_r[0] + 1j * curr_r[1])

    texto_rho = (
        r"$\hat{\rho}(t) = $" + "\n\n"
        f"[{rho_00:.3f} + 0.000j    {rho_01.real:+.3f} {rho_01.imag:+.3f}j]\n"
        f"[{rho_10.real:+.3f} {rho_10.imag:+.3f}j    {rho_11:.3f} + 0.000j]"
    )
    matrix_text.set_text(texto_rho)

    return shaft_line,

ani = FuncAnimation(fig, update, frames=frames_total, interval=33, blit=False)

try:
    writer = PillowWriter(fps=30)
    ani.save("simulacion_rabi_lindblad.gif", writer=writer)
except Exception as e:
    pass

plt.show()
