import numpy as np
import scipy.linalg
from scipy.integrate import solve_ivp, simpson
import multiprocessing as mp
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- 1. Efficient Dynamics Engine ---
def compute_path_length(args):
    N, theta_deg, t_span = args
    L_total, M_total, g = 1.0, 1.0, 9.8
    t_eval = np.linspace(0, t_span, 2000) # Resolution for path integration
    
    # Pre-calculate physical coefficients
    masses = np.full(N, M_total / N)
    lengths = np.full(N, L_total / N)
    m_suffix_sum = np.flip(np.cumsum(np.flip(masses)))
    
    L_grid = np.outer(lengths, lengths)
    M_coeffs = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            M_coeffs[i, j] = m_suffix_sum[max(i, j)] * L_grid[i, j]
    V_coeffs = m_suffix_sum * g * lengths

    def equations(t, z):
        p, q = z[:N], z[N:]
        diff_q = q[:, None] - q[None, :]
        M_num = M_coeffs * np.cos(diff_q) + np.eye(N) * 1e-11 
        
        try:
            # Use Cholesky for O(N^3) speedup in multi-link systems
            c, low = scipy.linalg.cho_factor(M_num, lower=True)
            q_dot = scipy.linalg.cho_solve((c, low), p)
        except:
            q_dot = np.linalg.solve(M_num, p)
            
        term1 = q_dot * np.sum(M_coeffs * np.sin(-diff_q) * q_dot, axis=1)
        return np.concatenate([term1 - V_coeffs * np.sin(q), q_dot])

    try:
        z0 = np.concatenate([np.zeros(N), np.full(N, np.radians(theta_deg))])
        sol = solve_ivp(equations, [0, t_span], z0, method='DOP853', 
                        t_eval=t_eval, rtol=1e-8, atol=1e-10)
        
        # Calculate velocity magnitude at the tip
        q_mat = sol.y[N:]
        x_tip = np.sum(lengths[:, None] * np.sin(q_mat), axis=0)
        y_tip = np.sum(-lengths[:, None] * np.cos(q_mat), axis=0)
        
        v_mag = np.sqrt(np.gradient(x_tip, sol.t)**2 + np.gradient(y_tip, sol.t)**2)
        path = simpson(v_mag, x=sol.t)
    except:
        path = np.nan
    
    return N, theta_deg, path

# --- 2. Grid Scan Execution ---
if __name__ == "__main__":
    # Define Grid Resolution
    N_list = np.arange(2, 101, 4)       # N from 2 to 100
    Theta_list = np.linspace(5, 90, 25) # Angles from 5 to 90 degrees
    T_SIM = 200                         # Match report duration
    
    tasks = [(n, th, T_SIM) for th in Theta_list for n in N_list]
    
    print(f"Generating FIG 17: {len(tasks)} simulation points...")
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = list(tqdm(pool.imap(compute_path_length, tasks), total=len(tasks)))

    # --- 3. Data Transformation & Heatmap Plotting ---
    N_grid, T_grid = np.meshgrid(N_list, Theta_list)
    P_grid = np.zeros_like(N_grid, dtype=float)
    
    # Map results back to grid
    for n, th, p in results:
        i = np.where(Theta_list == th)[0][0]
        j = np.where(N_list == n)[0][0]
        P_grid[i, j] = p

    plt.figure(figsize=(10, 8))
    
    # Create Contour Plot (matching FIG 17's aesthetic)
    levels = 20
    contour = plt.contourf(N_grid, T_grid, P_grid, levels=levels, cmap='jet')
    plt.contour(N_grid, T_grid, P_grid, levels=levels, colors='black', linewidths=0.2, alpha=0.5)
    
    # Label specific contour lines for clarity
    plt.clabel(plt.contour(N_grid, T_grid, P_grid, levels=[150, 280, 400]), inline=True, fontsize=8, fmt='%.1f')

    # Formatting
    plt.title('FIG. 17: Path Length Heatmap: $\\theta_0$ vs $N$', fontsize=14, pad=15)
    plt.xlabel('Number of Segments ($N$)', fontsize=12)
    plt.ylabel('Initial Angle $\\theta_0$ (deg)', fontsize=12)
    
    cbar = plt.colorbar(contour)
    cbar.set_label('Total Path Length ($P$)', rotation=270, labelpad=15)
    
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()