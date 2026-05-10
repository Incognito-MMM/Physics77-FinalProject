import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# --- 1. Simulation Setup ---
L_TOTAL, M_TOTAL, G = 1.0, 1.0, 9.8
T_SPAN = 200
FIXED_THETA = 5.0 

def equations_stable(t, z, N, m_suffix, W, L0, k, c):
    r, theta = z[:N], z[N:2*N]
    dr, dtheta = z[2*N:3*N], z[3*N:4*N]
    dT = theta[:, None] - theta[None, :]
    cos_dT, sin_dT = np.cos(dT), np.sin(dT)
    r_safe = np.maximum(r, L0 * 0.1)
    
    # Mass Matrix
    Mrr = W * cos_dT
    Mtt = W * (r_safe[:, None] @ r_safe[None, :]) * cos_dT
    Mrt = W * r_safe[None, :] * np.sin(-dT)
    M = np.block([[Mrr, Mrt], [Mrt.T, Mtt]])
    
    # Forces
    F_spring = -k * (r - L0) - c * dr
    Gr = m_suffix * G * np.cos(theta)
    V_t2, V_tr = dtheta[:, None] * dtheta[None, :], dtheta[:, None] * dr[None, :]

    Fr = F_spring + Gr + np.sum(W * r_safe[None, :] * V_t2 * cos_dT, axis=1) + 2 * np.sum(W * V_tr * sin_dT, axis=1)
    Gt_torque = -m_suffix * G * np.sin(theta) * r_safe
    Cor_torque_t = np.sum(W * r_safe[:, None] * r_safe[None, :] * V_t2 * sin_dT, axis=1) - 2 * m_suffix * r_safe * dr * dtheta 
    
    acc = np.linalg.solve(M + np.eye(2*N) * 1e-7, np.concatenate([Fr, Gt_torque + Cor_torque_t]))
    return np.concatenate([dr, dtheta, acc])

def run_task(N):
    L0, m = L_TOTAL / N, M_TOTAL / N
    k_val, c_val = 20000, 0.5 
    m_vec = np.full(N, m)
    m_suffix = np.flip(np.cumsum(np.flip(m_vec)))
    W = np.zeros((N, N))
    for i in range(N):
        for j in range(N): W[i, j] = m_suffix[max(i, j)]

    r_init = L0 + (m_suffix * G * np.cos(np.radians(FIXED_THETA))) / k_val
    z0 = np.concatenate([r_init, np.full(N, np.radians(FIXED_THETA)), np.zeros(2*N)])
    
    try:
        sol = solve_ivp(equations_stable, [0, T_SPAN], z0, method='Radau', rtol=1e-5, atol=1e-7, args=(N, m_suffix, W, L0, k_val, c_val))
        x = np.sum(sol.y[:N, :] * np.sin(sol.y[N:2*N, :]), axis=0)
        y = -np.sum(sol.y[:N, :] * np.cos(sol.y[N:2*N, :]), axis=0)
        return np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
    except: return np.nan

# --- 2. Main Execution (Restricted Range) ---
if __name__ == "__main__":
    # Restricted sampling for the fit: N = 5 to 20
    N_list = np.array([5, 8, 10, 12, 14, 16, 18, 20])
    
    print(f"Simulating Path Length for N in range [5, 20]...")
    with Pool(processes=cpu_count()) as pool:
        p_list = list(tqdm(pool.imap(run_task, N_list), total=len(N_list)))
    
    N_fit = N_list[~np.isnan(p_list)]
    P_fit = np.array(p_list)[~np.isnan(p_list)]

    # --- 3. Exponential Fitting ---
    def exp_func(x, A1, t1, y0):
        return A1 * np.exp(-x / t1) + y0

    popt, pcov = curve_fit(exp_func, N_fit, P_fit, p0=[-5, 5, P_fit[-1]])
    A1, t1, y0 = popt
    perr = np.sqrt(np.diag(pcov))

    # --- 4. Plotting ---
    plt.figure(figsize=(9, 6))
    n_smooth = np.linspace(5, 20, 100)
    plt.plot(n_smooth, exp_func(n_smooth, *popt), 'r-', label='Exponential Fit (N=5-20)')
    plt.plot(N_fit, P_fit, 'ks', markersize=6, label='Simulated Data')

    stats = (f"Model: ExpDec1\n"
             f"y0 (Limit): {y0:.4f} ± {perr[2]:.4f}\n"
             f"A1: {A1:.4f} ± {perr[0]:.4f}\n"
             f"t1: {t1:.4f} ± {perr[1]:.4f}")
    
    plt.gca().text(0.55, 0.15, stats, transform=plt.gca().transAxes, 
                   bbox=dict(facecolor='white', alpha=0.9), family='monospace')

    plt.xlabel('Number of Segments (N)')
    plt.ylabel('Total Path Length P (m)')
    plt.title('FIG. 20: Path Length Convergence (Restricted Fit: N=5-20)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    print(f"\nFinal Continuum Limit Estimate (y0): {y0:.4f} m")