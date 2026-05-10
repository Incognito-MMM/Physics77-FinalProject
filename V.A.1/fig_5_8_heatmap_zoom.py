import numpy as np
import scipy.linalg
import scipy.integrate
import sympy
import matplotlib.pyplot as plt
from matplotlib import colors
import multiprocessing
from functools import partial
from tqdm import tqdm

# --- 1. Physics Setup (Inherited from previous logic) ---
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

def compute_point(coords, max_t=6400):
    t0_deg, dt_deg = coords
    z_base = np.array([0, 0, np.deg2rad(t0_deg), np.deg2rad(t0_deg)])
    z_pert = np.array([0, 0, np.deg2rad(t0_deg + dt_deg), np.deg2rad(t0_deg + dt_deg)])
    
    def ode(t, Z):
        return np.concatenate([Hamilton_equations(t, Z[:4]), Hamilton_equations(t, Z[4:])])
    
    def event(t, Z): return np.linalg.norm(Z[2:4] - Z[6:8]) - 0.5
    event.terminal = True
    
    sol = scipy.integrate.solve_ivp(ode, [0, max_t], np.concatenate([z_base, z_pert]), 
                                    events=event, method='DOP853', rtol=1e-5, atol=1e-7)
    return sol.t_events[0][0] if sol.t_events[0].size > 0 else max_t

# --- 2. Zoom Plotter ---
def run_zoom_scan(theta_range, dev_range, res, title_prefix):
    thetas = np.linspace(theta_range[0], theta_range[1], res)
    devs = np.linspace(dev_range[0], dev_range[1], res)
    tasks = [(t, d) for t in thetas for d in devs]
    
    with multiprocessing.Pool() as pool:
        results = list(tqdm(pool.imap(partial(compute_point, max_t=6400), tasks), total=len(tasks), desc=title_prefix))
    
    data = np.array(results).reshape(res, res).T
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Linear View
    cmap_lin = plt.cm.viridis.copy(); cmap_lin.set_over('yellow')
    im1 = ax1.imshow(data, extent=[*theta_range, *dev_range], origin='lower', cmap=cmap_lin, aspect='auto', vmax=6399)
    plt.colorbar(im1, ax=ax1, extend='max').set_label('Time to Chaos (s)')
    ax1.set_title(f"{title_prefix} (Linear)\nStable: {np.sum(data>=6400)/data.size:.1%}")
    ax1.set_xlabel("$\\theta_0$ (deg)"); ax1.set_ylabel("$\\delta\\theta$ (deg)")

    # Log View
    log_data = data.copy(); log_data[log_data >= 6400] = np.nan
    cmap_log = plt.cm.magma.copy(); cmap_log.set_bad('white')
    im2 = ax2.imshow(log_data, extent=[*theta_range, *dev_range], origin='lower', cmap=cmap_log, aspect='auto', norm=colors.LogNorm(vmin=1, vmax=6400))
    plt.colorbar(im2, ax=ax2).set_label('Log Time (s)')
    ax2.set_title(f"{title_prefix} (Log)\nWhite = Stable > 6400s")
    ax2.set_xlabel("$\\theta_0$ (deg)")
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Fig 5 Zoom: Transition Boundary
    run_zoom_scan(theta_range=(0, 40), dev_range=(0, 1.0), res=200, title_prefix="FIG. 5 Zoom")
    
    # Fig 8 Zoom: Stability Bulge (Note the low delta_theta range)
    run_zoom_scan(theta_range=(0, 90), dev_range=(0, 0.1), res=200, title_prefix="FIG. 8 Zoom")