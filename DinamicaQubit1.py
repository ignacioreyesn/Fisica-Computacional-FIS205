import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection 

def definir_campos(t):
    Omega_0 = 5.0  
    Omega_1 = 0.5  
    omega = 5.0 
    
    Omega_x = Omega_1 * np.cos(omega * t)
    Omega_y = Omega_1 * np.sin(omega * t) 
    Omega_z = Omega_0
    
    return np.array([Omega_x, Omega_y, Omega_z])

t_max = 15                 
phi = np.pi / 4
theta = np.pi / 6
estado_inicial = [np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)] 

def bloch_solver_wrapper(t, r):
    Omega_vector = definir_campos(t)
    return np.cross(Omega_vector, r)

fps = 30
frames_total = int(t_max * fps)
t_eval = np.linspace(0, t_max, frames_total)

sol = solve_ivp(
    bloch_solver_wrapper, 
    [0, t_max], 
    estado_inicial,
    t_eval=t_eval, 
    rtol=1e-9, atol=1e-9
)
r_data = sol.y

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

u = np.linspace(0, 2 * np.pi, 50)
v = np.linspace(0, np.pi, 25)
ax.plot_wireframe(np.outer(np.cos(u), np.sin(v)), np.outer(np.sin(u), np.sin(v)), 
                  np.outer(np.ones(np.size(u)), np.cos(v)), color="gray", alpha=0.15, linewidth=0.8)
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

ax.view_init(elev=15, azim=30)
ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1])

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

    campos_actuales = definir_campos(t_eval[frame])
    intensidad = np.linalg.norm(campos_actuales)
    
    z_clamped = np.clip(curr_r[2], -1, 1)
    theta_deg = np.degrees(np.arccos(z_clamped))
    phi_deg = np.degrees(np.arctan2(curr_r[1], curr_r[0]))

    title_text.set_text(f"Tiempo: {t_eval[frame]:.2f}s | $|\Omega|$: {intensidad:.2f}\n"
                        f"$\\theta$: {theta_deg:.0f}° | $\\phi$: {phi_deg:.0f}°")

    return shaft_line,

ani = FuncAnimation(fig, update, frames=frames_total, interval=33, blit=False)

try:
    writer = PillowWriter(fps=30)
    ani.save("simulacion_rabi.gif", writer=writer)
except Exception as e:
    print(f"Error: {e}")

plt.show()

