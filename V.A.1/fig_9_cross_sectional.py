import numpy as np
import scipy.linalg
import scipy.integrate
import sympy
import matplotlib.pyplot as plt
from matplotlib import colors
import multiprocessing
from functools import partial
from tqdm import tqdm
import time

# --- 1. Physics Engine Construction ---
def setup_physics(N=2):
    """
    Sets up the Hamiltonian dynamics for an N-link rigid pendulum.
    Returns functions for Mass Matrix (Mf), its Gradient (dMf), and Potential Gradient (dVf).
    """
    g = 9.8
    theta = sympy.symbols(f'theta1:{N+1}', real=True)
    
    # Kinematic matrices for multi-body links
    B_inv = np.diag(np.ones(N)) - np.diag(np.ones(N-1), k=-1)
    B_mat = sympy.Matrix(scipy.linalg.inv(B_inv))
    L_mat = sympy.Matrix(np.diag(np.ones(N)))
    M_diag = sympy.Matrix(np.diag(np.ones(N)))
    
    # Vectorized trig functions for efficiency
    cos_q = sympy.Matrix([sympy.cos(t) for t in theta])
    sin_q = sympy.Matrix([sympy.sin(t) for t in theta])
    
    # Mass Matrix M(q)
    Cos_Mat, Sin_Mat = sympy.diag(*cos_q), sympy.diag(*sin_q)
    M_expr = (Cos_Mat @ L_mat @ B_mat.T @ M_diag @ B_mat @ L_mat @ Cos_Mat + 
              Sin_Mat @ L_mat @ B_mat.T @ M_diag @ B_mat @ L_mat @ Sin_Mat)
    
    # Potential Energy V(q)
    V_expr = (-g * sympy.Matrix(np.ones(N)).T @ (B_mat @ (L_mat @ cos_q)))[0]
    
    # Lambdify for fast numerical evaluation
    Mf = sympy.lambdify([theta], M_expr, "numpy")
    dMf = sympy.lambdify([theta], sympy.Matrix([sympy.flatten(sympy.diff(M_expr, ti)) for ti in theta]), "numpy", cse=True)
    dVf = sympy.lambdify([theta], sympy.Matrix([sympy.diff(V_expr, ti) for ti in theta]), "numpy")
    return Mf, dMf, dVf, N

Mf, dMf, dVf, N_links = setup_physics(N=2)

def Hamilton_equations(t, z):
    """Computes the state derivatives [p_dot, q_dot]."""
    p, q = z[:N_links], z[N_links:]
    M_num = Mf(q)
    # Solve for velocities: q_dot = M^-1 * p
    q_dot = scipy.linalg.solve(M_num, p, assume_a='pos')
    
    dM_stack = dMf(q).reshape(N_links, N_links, N_links)
    dV = dVf(q).flatten()
    
    # p_dot = -dH/dq
    p_dot = 0.5 * np.einsum('j,ijk,k->i', q_dot, dM_stack, q_dot) - dV
    return np.concatenate([p_dot, q_dot])

def compute_tc(dev_deg, theta0_deg, max_t):
    """Simulates two nearby trajectories until divergence threshold alpha=0.5."""
    t0, dt = np.deg2rad(theta0_deg), np.deg2rad(dev_deg)
    z_base = np.concatenate([np.zeros(N_links), [t0]*N_links])
    z_pert = np.concatenate([np.zeros(N_links), [t0+dt]*N_links])
    
    def ode_system(t, Z):
        return np.concatenate([Hamilton_equations(t, Z[:2*N_links]), 
                               Hamilton_equations(t, Z[2*N_links:])])
    
    def chaos_event(t, Z):
        # Chaos detected when Euclidean distance in configuration space > 0.5 rad
        return np.linalg.norm(Z[N_links:2*N_links] - Z[3*N_links:4*N_links]) - 0.5
    chaos_event.terminal = True
    
    sol = scipy.integrate.solve_ivp(ode_system, [0, max_t], np.concatenate([z_base, z_pert]), 
                                    events=chaos_event, method='DOP853', rtol=1e-5, atol=1e-7)
    return sol.t_events[0][0] if sol.t_events[0].size > 0 else max_t

# --- 2. Main Execution and Visualization ---
if __name__ == '__main__':
    # Simulation Parameters
    FIXED_THETA = 15.0      # Evaluation angle (deg)
    T_MAX = 6400            # Observation horizon (s)
    RES_DEV = 300           # Number of sampling points
    # Sampling logarithmically is essential for accurate power-law fitting
    DEV_SAMPLES = np.logspace(np.log10(0.001), np.log10(2.0), RES_DEV)

    print(f"Running Scaling Scan at theta_0 = {FIXED_THETA} degrees...")
    
    with multiprocessing.Pool() as pool:
        results = list(tqdm(pool.imap(partial(compute_tc, theta0_deg=FIXED_THETA, max_t=T_MAX), 
                                      DEV_SAMPLES), total=RES_DEV))
    
    results = np.array(results)
    # Filter points that reached the T_MAX (non-chaotic or too slow) for fitting
    mask = results < T_MAX * 0.98
    valid_dev, valid_tc = DEV_SAMPLES[mask], results[mask]

    # Perform Linear Regression on Log-Log data: ln(Tc) = slope * ln(dev) + intercept
    slope, intercept = np.polyfit(np.log(valid_dev), np.log(valid_tc), 1)

    # --- 3. Plotting Results ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Panel 1: Linear Scale
    ax1.plot(DEV_SAMPLES, results, 'o', markersize=2, alpha=0.5, label='Simulation')
    ax1.set_xlabel(r'Deviation $\delta\theta$ (deg)')
    ax1.set_ylabel(r'Chaos Time $T_C$ (s)')
    ax1.set_title(f'Linear Scale (Fixed $\\theta_0={FIXED_THETA}^\circ$)')
    ax1.grid(True, alpha=0.3)

    # Panel 2: Log-Log Scale with Power-Law Fit
    ax2.loglog(DEV_SAMPLES, results, 'o', markersize=3, alpha=0.4, label='Data')
    ax2.loglog(valid_dev, np.exp(intercept) * valid_dev**slope, 'r-', linewidth=2,
               label=f'Fit: $T_C = e^{{{intercept:.2f}}} \\cdot \\delta\\theta^{{{slope:.2f}}}$')
    
    ax2.set_xlabel(r'Deviation $\delta\theta$ (deg)')
    ax2.set_ylabel(r'$T_C$ (s)')
    ax2.set_title('Log-Log Scaling Law Analysis')
    ax2.legend()
    ax2.grid(True, which="both", alpha=0.2)

    plt.suptitle(f"Chaos Divergence Scaling Analysis (N={N_links})", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    print(f"\nScaling Law Derived: T_c = exp({intercept:.4f}) * delta_theta^({slope:.4f})")