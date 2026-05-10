import numpy as np
import scipy.linalg
import scipy.integrate
import sympy
import matplotlib.pyplot as plt
from matplotlib import colors
import time
import multiprocessing
from functools import partial
from tqdm import tqdm

# --- 1. Hamiltonian ---
def setup_physics(N_masses=2):
    g = 9.8
    theta = sympy.symbols(f'theta1:{N_masses+1}', real=True)
    lengths = np.ones(N_masses)
    mass = np.ones(N_masses)
    
    # Assistant Matrices
    I = np.diag(np.ones(N_masses))
    A = np.diag(np.ones(N_masses-1), k=-1)
    B = scipy.linalg.inv(I - A)
    B_mat = sympy.Matrix(B)
    L_mat = sympy.Matrix(np.diag(lengths))
    M_diag = sympy.Matrix(np.diag(mass))
    
    cos_theta_vec = sympy.Matrix([sympy.cos(t) for t in theta])
    Cos_Theta_mat = sympy.diag(*[sympy.cos(t) for t in theta])
    Sin_Theta_mat = sympy.diag(*[sympy.sin(t) for t in theta])
    
    # Mass Matrix
    M_theta_mat = (Cos_Theta_mat @ L_mat @ B_mat.T @ M_diag @ B_mat @ L_mat @ Cos_Theta_mat + 
                   Sin_Theta_mat @ L_mat @ B_mat.T @ M_diag @ B_mat @ L_mat @ Sin_Theta_mat)
    
    # Potential Energy
    h_vec = B_mat @ (L_mat @ cos_theta_vec)
    V = - g * sympy.Matrix(mass).T @ h_vec
    V = V[0]
    
    # Derivative
    dM_flat = sympy.Matrix([sympy.flatten(sympy.diff(M_theta_mat, ti)) for ti in theta])
    dV_dtheta = sympy.Matrix([sympy.diff(V, ti) for ti in theta])
    
    Mf = sympy.lambdify([theta], M_theta_mat, "numpy")
    dMf = sympy.lambdify([theta], dM_flat, "numpy", cse=True)
    dVf = sympy.lambdify([theta], dV_dtheta, "numpy")
    return Mf, dMf, dVf, N_masses

M_f, dM_f, dV_f, N = setup_physics()

def Hamilton_equations(t, z, M_f, dM_f, dV_f):
    p, q = z[:N], z[N:]
    M_num = M_f(q)
    try:
        lu = scipy.linalg.lu_factor(M_num)
        q_dot = scipy.linalg.lu_solve(lu, p)
    except:
        q_dot = np.linalg.solve(M_num, p)
    
    dM_stack = dM_f(q).reshape(N, N, N)
    dV = dV_f(q).flatten()
    
    term1 = 0.5 * np.einsum('j,ijk,k->i', q_dot, dM_stack, q_dot)
    return np.concatenate([term1 - dV, q_dot])

def compute_point(coords, max_t):
    theta_0_deg, delta_theta_deg = coords
    theta_0 = np.deg2rad(theta_0_deg)
    delta_theta = np.deg2rad(delta_theta_deg)
    
    # Initial Conditions
    z_base = np.array([0.0, 0.0, theta_0, theta_0])
    z_pert = np.array([0.0, 0.0, theta_0 + delta_theta, theta_0 + delta_theta])

    def combined_ode(t, Z):
        dz1 = Hamilton_equations(t, Z[:2*N], M_f, dM_f, dV_f)
        dz2 = Hamilton_equations(t, Z[2*N:], M_f, dM_f, dV_f)
        return np.concatenate([dz1, dz2])

    # Chaos Detection
    def deviation_event(t, Z):
        # Euclidean distance
        diff = Z[N:2*N] - Z[3*N:4*N]
        return np.linalg.norm(diff) - 0.5
    
    deviation_event.terminal = True

    sol = scipy.integrate.solve_ivp(
        combined_ode, [0, max_t], np.concatenate([z_base, z_pert]),
        events=deviation_event, method='DOP853', rtol=1e-5, atol=1e-7
    )
    return sol.t_events[0][0] if sol.t_events[0].size > 0 else max_t

# --- 2. Plot ---
def plot_heatmaps(data, max_t, fig_num, axes_row):
    res_x, res_y = data.shape
    stable_points = np.sum(data >= max_t)
    stable_ratio = (stable_points / data.size) * 100

    # Linear Scale
    ax_lin = axes_row[0]
    cmap_lin = plt.cm.viridis.copy()
    cmap_lin.set_over('yellow')
    im1 = ax_lin.imshow(data.T, extent=[0, 180, 0, 5], origin='lower', 
                        cmap=cmap_lin, aspect='auto', vmax=max_t*0.99)
    plt.colorbar(im1, ax=ax_lin, extend='max').set_label('Time to Chaos (s)')
    ax_lin.set_title(f"FIG. {fig_num}: Linear Scale ($T_{{max}}={max_t}s$)\nStable: {stable_ratio:.1f}%")
    ax_lin.set_ylabel("Deviation $\\delta\\theta$ (deg)")
    ax_lin.set_xlabel("Starting Angle $\\theta_0$ (deg)")

    # Log Scale
    ax_log = axes_row[1]
    log_data = data.copy().T
    log_data[log_data >= max_t] = np.nan
    cmap_log = plt.cm.magma.copy()
    cmap_log.set_bad('white')
    norm = colors.LogNorm(vmin=max(0.1, np.nanmin(log_data)) if np.any(~np.isnan(log_data)) else 0.1, vmax=max_t)
    im2 = ax_log.imshow(log_data, extent=[0, 180, 0, 5], origin='lower', 
                        cmap=cmap_log, aspect='auto', norm=norm)
    plt.colorbar(im2, ax=ax_log).set_label('Time to Chaos (log s)')
    ax_log.set_title(f"FIG. {fig_num}: Log Scale\nWhite = Stable beyond {max_t}s")
    ax_log.set_xlabel("Starting Angle $\\theta_0$ (deg)")

# --- 3. Scanning ---
if __name__ == '__main__':
    res = 60  # res = 60x60
    thetas = np.linspace(0, 180, res)
    devs = np.linspace(0, 5, res)
    tasks = [(t, d) for t in thetas for d in devs]
    
    fig, all_axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(hspace=0.3, wspace=0.2)

    for i, t_max in enumerate([100, 6400]):
        print(f"\n--- Running Scan for T_max = {t_max}s ---")
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            func = partial(compute_point, max_t=t_max)
            raw_results = list(tqdm(pool.imap(func, tasks), total=len(tasks)))
        
        heatmap_data = np.array(raw_results).reshape(res, res)
        plot_heatmaps(heatmap_data, t_max, i+2, all_axes[i])

    plt.suptitle("Double Pendulum Chaos Sensitivity Analysis", fontsize=16, y=0.95)
    plt.show()