import numpy as np
import scipy.linalg
import scipy.integrate
import sympy
import matplotlib.pyplot as plt
import multiprocessing
from functools import partial
from tqdm import tqdm

# --- 1. Physics Setup (Mass-Spring or Rigid) ---
def setup_physics(N_masses=2):
    g = 9.8
    theta = sympy.symbols(f'theta1:{N_masses+1}', real=True)
    lengths = np.ones(N_masses); mass = np.ones(N_masses)
    B_mat = sympy.Matrix(scipy.linalg.inv(np.diag(np.ones(N_masses)) - np.diag(np.ones(N_masses-1), k=-1)))
    L_mat = sympy.Matrix(np.diag(lengths)); M_diag = sympy.Matrix(np.diag(mass))
    cos_theta_vec = sympy.Matrix([sympy.cos(t) for t in theta])
    Cos_Theta_mat = sympy.diag(*[sympy.cos(t) for t in theta])
    Sin_Theta_mat = sympy.diag(*[sympy.sin(t) for t in theta])
    M_theta_mat = (Cos_Theta_mat @ L_mat @ B_mat.T @ M_diag @ B_mat @ L_mat @ Cos_Theta_mat + 
                   Sin_Theta_mat @ L_mat @ B_mat.T @ M_diag @ B_mat @ L_mat @ Sin_Theta_mat)
    V = (-g * sympy.Matrix(mass).T @ (B_mat @ (L_mat @ cos_theta_vec)))[0]
    Mf = sympy.lambdify([theta], M_theta_mat, "numpy")
    dMf = sympy.lambdify([theta], sympy.Matrix([sympy.flatten(sympy.diff(M_theta_mat, ti)) for ti in theta]), "numpy", cse=True)
    dVf = sympy.lambdify([theta], sympy.Matrix([sympy.diff(V, ti) for ti in theta]), "numpy")
    return Mf, dMf, dVf

M_f, dM_f, dV_f = setup_physics()

def Hamilton_equations(t, z):
    p, q = z[:2], z[2:]
    M_num = M_f(q)
    q_dot = scipy.linalg.solve(M_num, p, assume_a='pos')
    dM_stack = dM_f(q).reshape(2, 2, 2)
    dV = dV_f(q).flatten()
    p_dot = 0.5 * np.einsum('j,ijk,k->i', q_dot, dM_stack, q_dot) - dV
    return np.concatenate([p_dot, q_dot])

def compute_tc(theta0_deg, dev_deg, max_t=6400):
    z_base = np.array([0, 0, np.deg2rad(theta0_deg), np.deg2rad(theta0_deg)])
    z_pert = np.array([0, 0, np.deg2rad(theta0_deg + dev_deg), np.deg2rad(theta0_deg + dev_deg)])
    
    def ode(t, Z):
        return np.concatenate([Hamilton_equations(t, Z[:4]), Hamilton_equations(t, Z[4:])])
    
    def event(t, Z): return np.linalg.norm(Z[2:4] - Z[6:8]) - 0.5
    event.terminal = True
    
    sol = scipy.integrate.solve_ivp(ode, [0, max_t], np.concatenate([z_base, z_pert]), 
                                    events=event, method='DOP853', rtol=1e-5, atol=1e-7)
    return sol.t_events[0][0] if sol.t_events[0].size > 0 else max_t

# --- 2. Main Execution ---
if __name__ == '__main__':
    # Fixed deviations to analyze (matching the legend in your provided FIG. 7)
    fixed_devs = [3.0, 0.8, 0.2, 0.1, 0.05, 0.02]
    theta0_range = np.linspace(0, 175, 600) # High resolution for smooth curves
    
    plt.figure(figsize=(10, 7))
    colors = ['#555555', '#ff4d4d', '#1a75ff', '#2eb872', '#a64dff', '#ffcc00']
    markers = ['s', 'o', '^', 'v', 'd', '>']

    for i, dev in enumerate(fixed_devs):
        print(f"Processing cross-section for delta_theta = {dev}...")
        
        # Parallel processing for each curve
        with multiprocessing.Pool() as pool:
            func = partial(compute_tc, dev_deg=dev)
            tc_values = list(tqdm(pool.imap(func, theta0_range), total=len(theta0_range), leave=False))
        
        # Plot with markers (sampling markers for clarity)
        plt.plot(theta0_range, tc_values, label=f'Dev_{dev}', 
                 color=colors[i], marker=markers[i], markersize=6, alpha=0.8)

    # --- 3. Formatting ---
    plt.xlabel(r'$\theta_0$ (deg)', fontsize=14)
    plt.ylabel(r'$T_C$ (s)', fontsize=14)
    plt.title(r'Cross-sectional Analysis: $T_C$ vs $\theta_0$ for Various Fixed $\delta\theta$', fontsize=15)
    plt.legend(loc='upper right', frameon=True, fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.ylim(-200, 7000) # Leave room for the stable plateau at 6400s
    plt.xlim(-5, 180)
    
    # Save or show
    plt.tight_layout()
    plt.show()