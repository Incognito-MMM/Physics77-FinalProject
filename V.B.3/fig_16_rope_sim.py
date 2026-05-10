import numpy as np
import scipy.linalg
from scipy.integrate import solve_ivp, simpson
import multiprocessing as mp
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- 1. Core Simulation Engine ---
def run_convergence_sim(args):
    N, theta_deg, t_span = args
    
    # Physics constants (Dimensionless / Normalized)
    L_total, M_total, g = 1.0, 1.0, 9.8
    t_eval = np.linspace(0, t_span, 4000) 
    
    # Pre-calculate Mass and Potential matrices for N-link pendulum
    masses = np.full(N, M_total / N)
    lengths = np.full(N, L_total / N)
    m_suffix_sum = np.flip(np.cumsum(np.flip(masses)))
    
    # M_coeffs[i,j] accounts for the mass supported by the higher-order joints
    L_grid = np.outer(lengths, lengths)
    M_coeffs = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            M_coeffs[i, j] = m_suffix_sum[max(i, j)] * L_grid[i, j]
    V_coeffs = m_suffix_sum * g * lengths

    def equations(t, z):
        p, q = z[:N], z[N:]
        diff_q = q[:, None] - q[None, :]
        # Inertia Matrix M(q)
        M_num = M_coeffs * np.cos(diff_q) + np.eye(N) * 1e-11 # Regularization
        
        try:
            # Cholesky decomposition for faster linear solving
            c, low = scipy.linalg.cho_factor(M_num, lower=True)
            q_dot = scipy.linalg.cho_solve((c, low), p)
        except:
            q_dot = np.linalg.solve(M_num, p)
            
        # Coriolis and Centripetal terms
        term1 = q_dot * np.sum(M_coeffs * np.sin(-diff_q) * q_dot, axis=1)
        return np.concatenate([term1 - V_coeffs * np.sin(q), q_dot])

    try:
        # Initial conditions: zero momentum, all segments at theta_deg
        z0 = np.concatenate([np.zeros(N), np.full(N, np.radians(theta_deg))])
        sol = solve_ivp(equations, [0, t_span], z0, method='DOP853', 
                        t_eval=t_eval, rtol=1e-8, atol=1e-10)
        
        # Calculate Tip Trajectory (End-effector position)
        q_mat = sol.y[N:]
        x_tip = np.sum(lengths[:, None] * np.sin(q_mat), axis=0)
        y_tip = np.sum(-lengths[:, None] * np.cos(q_mat), axis=0)
        
        # Velocity magnitude for path integral
        vx = np.gradient(x_tip, sol.t)
        vy = np.gradient(y_tip, sol.t)
        v_mag = np.sqrt(vx**2 + vy**2)
        
        # P = integral of speed over time
        path_length = simpson(v_mag, x=sol.t)
    except Exception:
        path_length = np.nan
    
    return theta_deg, N, path_length

# --- 2. Main Execution and Visualization ---
if __name__ == "__main__":
    # Parameters matching FIG. 16
    theta_list = [5, 15, 30, 90]
    # N values from discrete (2-link) to continuum approximation (200-link)
    n_values = [2, 4, 8, 16, 32, 50, 100, 200]
    t_sim = 200 
    
    tasks = [(n, th, t_sim) for th in theta_list for n in n_values]
    
    print(f"Starting Convergence Scan for Theta: {theta_list}")
    results = []
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = list(tqdm(pool.imap(run_convergence_sim, tasks), total=len(tasks)))

    # Organising data for plotting
    data = {th: {'n': [], 'p': []} for th in theta_list}
    for th, n, p in results:
        data[th]['n'].append(n)
        data[th]['p'].append(p)

    # --- 3. Plotting FIG. 16 ---
    plt.figure(figsize=(12, 8))
    colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd'] # Blue, Green, Red, Purple
    
    for i, th in enumerate(theta_list):
        n_arr = np.array(data[th]['n'])
        p_arr = np.array(data[th]['p'])
        
        # Sort by N for plotting
        sort_idx = np.argsort(n_arr)
        n_plot, p_plot = n_arr[sort_idx], p_arr[sort_idx]
        
        # Simulation lines
        plt.plot(n_plot, p_plot, marker='o', markersize=4, label=f'Simulation ({th}°)', color=colors[i])
        
        # Horizontal Asymptote (Theoretical/Continuum Limit)
        asymptote = p_plot[-1] 
        plt.axhline(y=asymptote, color=colors[i], linestyle='--', alpha=0.6)
        plt.text(180, asymptote * 1.02, f'Asy: {asymptote:.2f}', color=colors[i], fontsize=9)

    # Formatting
    plt.yscale('log') # Path length is visualized on a log scale
    plt.xlabel('Number of Segments (N)', fontsize=12)
    plt.ylabel('Total Path Length (m) [Log Scale]', fontsize=12)
    plt.title('FIG. 16: Convergence of Path Length (P) to N', fontsize=14, pad=20)
    plt.legend(loc='upper right')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.xlim(0, 210)
    
    plt.tight_layout()
    plt.show()