import numpy as np
import scipy.linalg
import scipy.integrate
import sympy
import matplotlib.pyplot as plt
from matplotlib import colors
import time
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
import os
import warnings
import pickle
warnings.filterwarnings('ignore')

# --- Configuration ---
N_masses = 3
g = 9.8
CHAOS_THRESHOLD = 0.05  # radians
t_span = 1800

# Grid parameters
res_x, res_y = 100, 100
theta_range = [0, 40]
dev_range = [0, 1.0]

# Optimized settings
CHECKPOINT_INTERVAL = 500  # Save less frequently
NUM_CORES = mp.cpu_count()  # Use ALL cores
print(f"Detected {NUM_CORES} CPU cores - will use ALL of them")

# --- Global variables for the worker processes ---
M_f = None
dM_f = None
dV_f = None

def init_worker():
    """Initialize physics for each worker process (runs once per process)"""
    global M_f, dM_f, dV_f
    
    print("Generating symbolic physics equations in worker...")
    theta = sympy.symbols(f'theta1:{N_masses+1}', real=True)
    lengths = np.ones(N_masses)
    mass = np.ones(N_masses)
    
    I = np.diag(np.ones(N_masses))
    A = np.diag(np.ones(N_masses-1), k=-1)
    B = scipy.linalg.inv(I - A)
    B_mat = sympy.Matrix(B)
    L_mat = sympy.Matrix(np.diag(lengths))
    M_diag = sympy.Matrix(np.diag(mass))

    cos_theta_vec = sympy.Matrix([sympy.cos(t) for t in theta])
    Cos_Theta_mat = sympy.diag(*[sympy.cos(t) for t in theta])
    Sin_Theta_mat = sympy.diag(*[sympy.sin(t) for t in theta])

    M_theta_mat = (Cos_Theta_mat @ L_mat @ B_mat.T @ M_diag @ B_mat @ L_mat @ Cos_Theta_mat + 
                   Sin_Theta_mat @ L_mat @ B_mat.T @ M_diag @ B_mat @ L_mat @ Sin_Theta_mat)

    h_vec = B_mat @ (L_mat @ cos_theta_vec)
    V = - g * sympy.Matrix(mass).T @ h_vec
    V = V[0]

    dM_flat = sympy.Matrix([sympy.flatten(sympy.diff(M_theta_mat, ti)) for ti in theta])
    dV_dtheta = sympy.Matrix([sympy.diff(V, ti) for ti in theta])

    M_f = sympy.lambdify([theta], M_theta_mat, "numpy")
    dM_f = sympy.lambdify([theta], dM_flat, "numpy", cse=True)
    dV_f = sympy.lambdify([theta], dV_dtheta, "numpy")
    
    print("Worker physics ready!")

def hamilton_equations(t, z):
    """Hamilton's equations for triple pendulum"""
    global M_f, dM_f, dV_f
    
    p, q = z[:N_masses], z[N_masses:]
    M_num = M_f(q)
    
    try:
        q_dot = np.linalg.solve(M_num, p)
    except:
        q_dot = np.linalg.lstsq(M_num, p, rcond=None)[0]
    
    dM_stack = dM_f(q).reshape(N_masses, N_masses, N_masses)
    dV = dV_f(q).flatten()
    term1 = 0.5 * np.einsum('j,ijk,k->i', q_dot, dM_stack, q_dot)
    
    return np.concatenate([term1 - dV, q_dot])

def get_chaos_time(args):
    """Calculate chaos time for a single simulation"""
    base_theta_deg, deviation_deg = args
    
    # Quick exit for zero deviation
    if deviation_deg == 0:
        return t_span
    
    base_theta = np.deg2rad(base_theta_deg)
    dev_theta = np.deg2rad(deviation_deg)
    
    z_base = np.array([0.0, 0.0, 0.0, base_theta, base_theta, base_theta])
    z_pert = np.array([0.0, 0.0, 0.0, base_theta + dev_theta, 
                       base_theta + dev_theta, base_theta + dev_theta])

    def combined_ode(t, Z):
        dz1 = hamilton_equations(t, Z[:6])
        dz2 = hamilton_equations(t, Z[6:])
        return np.concatenate([dz1, dz2])

    def deviation_event(t, Z, *args):
        dist = np.linalg.norm(Z[3:6] - Z[9:12])
        return dist - CHAOS_THRESHOLD
    deviation_event.terminal = True

    try:
        sol = scipy.integrate.solve_ivp(
            combined_ode, [0, t_span], 
            np.concatenate([z_base, z_pert]),
            events=deviation_event,
            method='DOP853',  # Faster than RK45 for this problem
            rtol=1e-5,  # Slightly relaxed tolerance
            atol=1e-7,
            max_step=0.2  # Larger steps for speed
        )
        
        if sol.t_events[0].size > 0:
            return sol.t_events[0][0]
    except:
        pass
    
    return t_span

def main():
    print("="*70)
    print("🚀 OPTIMIZED TRIPLE PENDULUM CHAOS ANALYSIS")
    print("="*70)
    print(f"CPU Cores Available: {NUM_CORES}")
    print(f"Grid: {res_x} x {res_y} = {res_x * res_y} simulations")
    print(f"Max time per simulation: {t_span} seconds")
    print("="*70)
    
    # Create parameter grid
    initial_angles = np.linspace(theta_range[0], theta_range[1], res_x)
    angle_deviations = np.linspace(dev_range[0], dev_range[1], res_y)
    
    # Create all tasks
    tasks = [(theta, dev) for theta in initial_angles for dev in angle_deviations]
    
    print(f"\n🚀 Starting parallel processing with {NUM_CORES} workers...")
    print("Each worker will generate physics equations once (this takes ~10 seconds each)")
    print("Then they'll run simulations in parallel at full speed!\n")
    
    start_time = time.time()
    
    # Run parallel processing with initialization
    with mp.Pool(processes=NUM_CORES, initializer=init_worker) as pool:
        results = list(tqdm(
            pool.imap(get_chaos_time, tasks, chunksize=10),  # Process in chunks
            total=len(tasks),
            desc="Computing chaos map",
            unit="sim"
        ))
    
    # Reshape results
    chaos_times = np.array(results).reshape(res_x, res_y)
    
    total_time = time.time() - start_time
    print(f"\n✅ All simulations complete!")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Average per simulation: {total_time/len(tasks):.2f} seconds")
    
    # --- Create heatmap ---
    print("\n📊 Creating heatmap...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Linear scale
    im1 = axes[0].imshow(chaos_times.T, 
                         extent=[theta_range[0], theta_range[1], 
                                dev_range[0], dev_range[1]],
                         origin='lower', aspect='auto', cmap='viridis',
                         vmin=0, vmax=t_span)
    axes[0].set_xlabel('Initial Angle (degrees)', fontsize=12)
    axes[0].set_ylabel('Deviation Angle (degrees)', fontsize=12)
    axes[0].set_title('Triple Pendulum Chaos Time (seconds)', fontsize=12)
    plt.colorbar(im1, ax=axes[0])
    
    # Log scale
    log_display = chaos_times.copy().astype(float)
    log_display[log_display >= t_span] = np.nan
    cmap_log = plt.cm.plasma.copy()
    cmap_log.set_bad(color='white')
    
    chaotic_times = chaos_times[chaos_times < t_span]
    min_time = chaotic_times.min() if len(chaotic_times) > 0 else 0.1
    
    im2 = axes[1].imshow(log_display.T, 
                         extent=[theta_range[0], theta_range[1], 
                                dev_range[0], dev_range[1]],
                         origin='lower', aspect='auto', cmap=cmap_log,
                         norm=colors.LogNorm(vmin=min_time, vmax=t_span))
    axes[1].set_xlabel('Initial Angle (degrees)', fontsize=12)
    axes[1].set_ylabel('Deviation Angle (degrees)', fontsize=12)
    axes[1].set_title('Log Scale (White = Stable)', fontsize=12)
    plt.colorbar(im2, ax=axes[1])
    
    plt.suptitle(f'Triple Pendulum Chaos Analysis\nThreshold: {np.rad2deg(CHAOS_THRESHOLD):.2f}°, Max Time: {t_span}s', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save everything
    results_dir = "triple_pendulum_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(results_dir, f'triple_pendulum_optimized_{timestamp}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ Heatmap saved: {filename}")
    
    data_filename = os.path.join(results_dir, f'optimized_data_{timestamp}.npz')
    np.savez(data_filename,
             chaos_times=chaos_times,
             initial_angles=initial_angles,
             angle_deviations=angle_deviations,
             threshold_rad=CHAOS_THRESHOLD,
             max_time=t_span)
    print(f"✅ Data saved: {data_filename}")
    
    # Statistics
    chaotic_points = np.sum(chaos_times < t_span)
    stable_points = np.sum(chaos_times >= t_span)
    
    print("\n" + "="*70)
    print("📊 RESULTS SUMMARY")
    print("="*70)
    print(f"Total simulations: {len(tasks)}")
    print(f"Chaotic (<{t_span}s): {chaotic_points} ({chaotic_points/len(tasks)*100:.1f}%)")
    print(f"Stable (={t_span}s): {stable_points} ({stable_points/len(tasks)*100:.1f}%)")
    
    if len(chaotic_times) > 0:
        print(f"\nChaos time statistics (chaotic region only):")
        print(f"  Minimum: {chaotic_times.min():.1f} seconds")
        print(f"  Maximum: {chaotic_times.max():.1f} seconds")
        print(f"  Mean: {chaotic_times.mean():.1f} seconds")
        print(f"  Median: {np.median(chaotic_times):.1f} seconds")
    
    plt.show()

if __name__ == "__main__":
    # Required for macOS multiprocessing
    mp.freeze_support()
    main()