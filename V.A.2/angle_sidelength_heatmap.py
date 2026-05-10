import numpy as np
import scipy.linalg
import scipy.integrate
import sympy
import matplotlib.pyplot as plt
from matplotlib import colors
import warnings
warnings.filterwarnings('ignore')

# Setup physics (runs once)
g = 9.8
t_span = 200

def setup_physics():
    theta = sympy.symbols('theta1 theta2', real=True)
    lengths = sympy.symbols('l1 l2', real=True)
    mass = np.ones(2)
    
    B = sympy.Matrix(np.linalg.inv(np.eye(2) - np.diag(np.ones(1), k=-1)))
    L_mat = sympy.Matrix([[lengths[0], 0], [0, lengths[1]]])
    M_diag = sympy.Matrix(np.diag(mass))
    
    cos_theta_vec = sympy.Matrix([sympy.cos(t) for t in theta])
    Cos_Theta_mat = sympy.diag(*[sympy.cos(t) for t in theta])
    Sin_Theta_mat = sympy.diag(*[sympy.sin(t) for t in theta])
    
    M_theta_mat = (Cos_Theta_mat @ L_mat @ B.T @ M_diag @ B @ L_mat @ Cos_Theta_mat + 
                   Sin_Theta_mat @ L_mat @ B.T @ M_diag @ B @ L_mat @ Sin_Theta_mat)
    
    h_vec = B @ (L_mat @ cos_theta_vec)
    V = - g * sympy.Matrix(mass).T @ h_vec
    
    dM_flat = sympy.Matrix([sympy.flatten(sympy.diff(M_theta_mat, ti)) for ti in theta])
    dV_dtheta = sympy.Matrix([sympy.diff(V[0], ti) for ti in theta])
    
    return (sympy.lambdify([theta, lengths], M_theta_mat, "numpy"),
            sympy.lambdify([theta, lengths], dM_flat, "numpy", cse=True),
            sympy.lambdify([theta, lengths], dV_dtheta, "numpy"))

M_f, dM_f, dV_f = setup_physics()

def Hamilton_equations(t, z, lengths):
    p, q = z[:2], z[2:]
    M_num = M_f(q, lengths)
    q_dot = np.linalg.solve(M_num, p)
    dM_stack = dM_f(q, lengths).reshape(2,2,2)
    dV = dV_f(q, lengths).flatten()
    term1 = 0.5 * np.einsum('j,ijk,k->i', q_dot, dM_stack, q_dot)
    return np.concatenate([term1 - dV, q_dot])

def chaos_time(base_deg, dev_deg, length_ratio, max_t=t_span):
    base_theta = np.deg2rad(base_deg)
    dev_theta = np.deg2rad(dev_deg)
    lengths = np.array([1.0, length_ratio])
    
    z_base = np.array([0.0, 0.0, base_theta, base_theta])
    z_pert = np.array([0.0, 0.0, base_theta + dev_theta, base_theta + dev_theta])
    
    def combined_ode(t, Z):
        return np.concatenate([Hamilton_equations(t, Z[:4], lengths),
                               Hamilton_equations(t, Z[4:], lengths)])
    
    def deviation_event(t, Z):
        return np.linalg.norm(Z[2:4] - Z[6:8]) - 0.5
    deviation_event.terminal = True
    
    try:
        sol = scipy.integrate.solve_ivp(combined_ode, [0, max_t], 
                                       np.concatenate([z_base, z_pert]),
                                       events=deviation_event, method='RK45',
                                       rtol=1e-5, max_step=0.1)
        if sol.t_events[0].size > 0:
            return sol.t_events[0][0]
    except:
        pass
    return max_t

# Generate data
base_angle = 50.0
angle_deviations = np.linspace(0.4, 3.0, 52)
length_ratios = np.linspace(1, 4, 100)

chaos_times = np.zeros((len(angle_deviations), len(length_ratios)))
for i, dev_angle in enumerate(angle_deviations):
    for j, ratio in enumerate(length_ratios):
        chaos_times[i, j] = chaos_time(base_angle, dev_angle, ratio)

# Plotting
fig = plt.figure(figsize=(15, 5))

# Heatmap
ax1 = plt.subplot(1, 3, 1)
im = ax1.imshow(chaos_times.T, extent=[angle_deviations[0], angle_deviations[-1], 
                                       length_ratios[0], length_ratios[-1]],
                origin='lower', aspect='auto', cmap='viridis', vmin=0, vmax=t_span)
ax1.set_xlabel('Angle Deviation (degrees)')
ax1.set_ylabel('Length Ratio (l₂/l₁)')
ax1.set_title('Chaos Time Heatmap')
plt.colorbar(im, ax=ax1, label='Seconds')

# Profiles at fixed length ratios
ax2 = plt.subplot(1, 3, 2)
selected_ratios = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
colors_line = plt.cm.viridis(np.linspace(0, 1, len(selected_ratios)))

for idx, ratio in enumerate(selected_ratios):
    ratio_idx = np.argmin(np.abs(length_ratios - ratio))
    ax2.plot(angle_deviations, chaos_times[:, ratio_idx], 
            color=colors_line[idx], linewidth=2, label=f'l₂/l₁ = {ratio:.1f}')
ax2.set_xlabel('Angle Deviation (degrees)')
ax2.set_ylabel('Chaos Time (seconds)')
ax2.set_title('Fixed Length Ratios')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, t_span)

# Profiles at fixed angle deviations
ax3 = plt.subplot(1, 3, 3)
selected_deviations = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
colors_line2 = plt.cm.plasma(np.linspace(0, 1, len(selected_deviations)))

for idx, dev in enumerate(selected_deviations):
    dev_idx = np.argmin(np.abs(angle_deviations - dev))
    ax3.plot(length_ratios, chaos_times[dev_idx, :], 
            color=colors_line2[idx], linewidth=2, label=f'{dev}°')
ax3.set_xlabel('Length Ratio (l₂/l₁)')
ax3.set_ylabel('Chaos Time (seconds)')
ax3.set_title('Fixed Angle Deviations')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)
ax3.set_ylim(0, t_span)

plt.suptitle(f'Double Pendulum Chaos (Base Angle = {base_angle}°)', fontsize=14)
plt.tight_layout()
plt.show()