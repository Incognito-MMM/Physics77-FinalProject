import numpy as np
import scipy.linalg
import scipy.integrate
import sympy
import matplotlib.pyplot as plt
from matplotlib import colors
import time
import warnings
import os
warnings.filterwarnings('ignore')

print("="*70)
print("WORKING DOUBLE PENDULUM CHAOS ANALYSIS")
print("="*70)

# Parameters - CHANGED FROM 3 TO 2 FOR DOUBLE PENDULUM
N_masses = 2  # ← CHANGED: 2 masses for double pendulum
g = 9.8
CHAOS_THRESHOLD = 0.05  # radians
t_span = 1800  # Maximum simulation time

# Grid parameters (same as triple pendulum)
res_x, res_y = 100, 100  # Full grid
theta_range = [0, 40]
dev_range = [0, 1.0]

print(f"Grid: {res_x} x {res_y} = {res_x * res_y} simulations")
print(f"Max time per simulation: {t_span} seconds")
print("Generating physics equations (this takes ~5 seconds for double pendulum)...")

# --- Create results directory for checkpoints ---
results_dir = "double_pendulum_results"  # ← CHANGED: separate folder for double pendulum
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    print(f"📁 Created results directory: {results_dir}")

# --- Check for existing progress ---
checkpoint_file = os.path.join(results_dir, "progress.npz")
chaos_times = np.zeros((res_x, res_y))
start_i, start_j = 0, 0
completed_so_far = 0

if os.path.exists(checkpoint_file):
    print("\n📂 Found existing checkpoint! Resuming...")
    data = np.load(checkpoint_file, allow_pickle=True)
    chaos_times = data['chaos_times']
    start_i = int(data['last_i'])
    start_j = int(data['last_j'])
    completed_so_far = int(data['completed'])
    print(f"✅ Resuming from row {start_i}, column {start_j}")
    print(f"   Already completed: {completed_so_far}/{res_x * res_y} simulations")
else:
    print("\n🆕 Starting fresh simulation...")

# --- Generate the DOUBLE pendulum equations ---
theta = sympy.symbols(f'theta1:{N_masses+1}', real=True)
lengths = np.ones(N_masses)
mass = np.ones(N_masses)

# Auxiliary matrix calculation (same method as triple pendulum)
I = np.diag(np.ones(N_masses))
A = np.diag(np.ones(N_masses-1), k=-1)
B = scipy.linalg.inv(I - A)
B_mat = sympy.Matrix(B)
L_mat = sympy.Matrix(np.diag(lengths))
M_diag = sympy.Matrix(np.diag(mass))

cos_theta_vec = sympy.Matrix([sympy.cos(t) for t in theta])
Cos_Theta_mat = sympy.diag(*[sympy.cos(t) for t in theta])
Sin_Theta_mat = sympy.diag(*[sympy.sin(t) for t in theta])

# Kinetic energy mass matrix M(theta)
M_theta_mat = (Cos_Theta_mat @ L_mat @ B_mat.T @ M_diag @ B_mat @ L_mat @ Cos_Theta_mat + 
               Sin_Theta_mat @ L_mat @ B_mat.T @ M_diag @ B_mat @ L_mat @ Sin_Theta_mat)

# Potential energy V
h_vec = B_mat @ (L_mat @ cos_theta_vec)
V = - g * sympy.Matrix(mass).T @ h_vec
V = V[0]

# Derivatives
dM_flat = sympy.Matrix([sympy.flatten(sympy.diff(M_theta_mat, ti)) for ti in theta])
dV_dtheta = sympy.Matrix([sympy.diff(V, ti) for ti in theta])

# Convert to numerical functions
M_f = sympy.lambdify([theta], M_theta_mat, "numpy")
dM_f = sympy.lambdify([theta], dM_flat, "numpy", cse=True)
dV_f = sympy.lambdify([theta], dV_dtheta, "numpy")

print("✅ Double pendulum physics equations generated!")

# --- Hamilton's equations ---
def hamilton_equations(t, z):
    p, q = z[:N_masses], z[N_masses:]
    
    # Get numerical matrices
    M_num = M_f(q)
    
    # Solve for q_dot = M^(-1) * p
    try:
        q_dot = np.linalg.solve(M_num, p)
    except:
        q_dot = np.linalg.lstsq(M_num, p, rcond=None)[0]
    
    # Calculate derivatives
    dM_stack = dM_f(q).reshape(N_masses, N_masses, N_masses)
    dV = dV_f(q).flatten()
    
    # d/dt (∂L/∂q̇) = ½ q̇ᵀ ∂M/∂q q̇ - ∂V/∂q
    term1 = 0.5 * np.einsum('j,ijk,k->i', q_dot, dM_stack, q_dot)
    
    return np.concatenate([term1 - dV, q_dot])

# --- Chaos detection function (adjusted for 2 masses) ---
def get_chaos_time(base_theta_deg, deviation_deg):
    base_theta = np.deg2rad(base_theta_deg)
    dev_theta = np.deg2rad(deviation_deg)
    
    # Skip if deviation is zero (no chaos possible)
    if dev_theta == 0:
        return t_span
    
    # Initial states - DOUBLE PENDULUM HAS 2 ANGLES, so 4 state variables total
    # State: [p1, p2, theta1, theta2] → 4 variables
    # For tracking chaos: we need base and perturbed trajectories
    z_base = np.array([0.0, 0.0, base_theta, base_theta])  # [p1, p2, θ1, θ2]
    z_pert = np.array([0.0, 0.0, base_theta + dev_theta, base_theta + dev_theta])

    def combined_ode(t, Z):
        # Z has 8 variables: [base_p1, base_p2, base_θ1, base_θ2, pert_p1, pert_p2, pert_θ1, pert_θ2]
        dz1 = hamilton_equations(t, Z[:4])
        dz2 = hamilton_equations(t, Z[4:])
        return np.concatenate([dz1, dz2])

    def deviation_event(t, Z, *args):
        # Track angular separation (angle coordinates only: indices 2,3 vs 6,7)
        dist = np.linalg.norm(Z[2:4] - Z[6:8])
        return dist - CHAOS_THRESHOLD
    deviation_event.terminal = True

    try:
        sol = scipy.integrate.solve_ivp(
            combined_ode, 
            [0, t_span], 
            np.concatenate([z_base, z_pert]),
            events=deviation_event,
            method='RK45', 
            rtol=1e-6, 
            atol=1e-8,
            max_step=0.1
        )

        if sol.t_events[0].size > 0:
            return sol.t_events[0][0]
        else:
            return t_span
    except Exception as e:
        return t_span

# --- Run simulations ---
print("\n🚀 Running simulations for DOUBLE PENDULUM...")
print("(This will be FASTER than triple pendulum - about 2-3x speedup)")
print("Progress will be saved every 50 simulations")
print("Press Ctrl+C to save and exit gracefully\n")

initial_angles = np.linspace(theta_range[0], theta_range[1], res_x)
angle_deviations = np.linspace(dev_range[0], dev_range[1], res_y)

start_time = time.time()
total_sims = res_x * res_y
completed = completed_so_far

# Track the last saved checkpoint
last_checkpoint = 0

for i in range(start_i, res_x):
    for j in range(start_j if i == start_i else 0, res_y):
        # Skip if already computed
        if chaos_times[i, j] != 0:
            continue
            
        theta = initial_angles[i]
        dev = angle_deviations[j]
        chaos_times[i, j] = get_chaos_time(theta, dev)
        completed += 1
        
        # --- SAVE CHECKPOINT EVERY 50 SIMULATIONS ---
        if completed % 50 == 0 and completed > last_checkpoint:
            np.savez(checkpoint_file, 
                     chaos_times=chaos_times, 
                     last_i=i, 
                     last_j=j,
                     completed=completed)
            last_checkpoint = completed
            
            # Also save a backup with timestamp
            backup_file = os.path.join(results_dir, f"backup_{completed}.npz")
            np.savez(backup_file,
                     chaos_times=chaos_times,
                     last_i=i,
                     last_j=j,
                     completed=completed)
            
            # Progress update
            elapsed = time.time() - start_time
            if completed > completed_so_far:
                sims_since_resume = completed - completed_so_far
                time_per_sim = elapsed / max(1, sims_since_resume)
                remaining_sims = total_sims - completed
                eta_seconds = remaining_sims * time_per_sim
                eta_hours = eta_seconds / 3600
                
                print(f"\n💾 Checkpoint saved at {completed}/{total_sims}")
                print(f"   Time per simulation: {time_per_sim:.1f} seconds")
                print(f"   Estimated time remaining: {eta_hours:.1f} hours")
        
        # Progress indicator (update every simulation)
        if completed % 10 == 0:
            progress = (completed / total_sims) * 100
            print(f"Progress: {progress:.1f}% | Completed: {completed}/{total_sims}", end="\r")

# --- Save final checkpoint ---
np.savez(checkpoint_file, 
         chaos_times=chaos_times, 
         last_i=res_x-1, 
         last_j=res_y-1,
         completed=total_sims)
print(f"\n\n💾 Final checkpoint saved!")

total_time = time.time() - start_time
print(f"\n✅ Simulations complete! Time: {total_time/3600:.2f} hours")

# --- Create heatmap ---
print("\n📊 Creating heatmap...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Heatmap 1: Linear scale
im1 = axes[0].imshow(chaos_times.T, 
                     extent=[theta_range[0], theta_range[1], 
                            dev_range[0], dev_range[1]],
                     origin='lower', aspect='auto', cmap='viridis',
                     vmin=0, vmax=t_span)
axes[0].set_xlabel('Initial Angle (degrees)', fontsize=12)
axes[0].set_ylabel('Deviation Angle (degrees)', fontsize=12)
axes[0].set_title('Double Pendulum Chaos Time (seconds)', fontsize=12)
cbar1 = plt.colorbar(im1, ax=axes[0])
cbar1.set_label('Time to Chaos (seconds)', fontsize=10)

# Heatmap 2: Log scale (white = stable)
log_display = chaos_times.copy().astype(float)
log_display[log_display >= t_span] = np.nan
cmap_log = plt.cm.plasma.copy()
cmap_log.set_bad(color='white')

# Find min non-stable time for proper scaling
chaotic_times = chaos_times[chaos_times < t_span]
min_time = chaotic_times.min() if len(chaotic_times) > 0 else 0.1

im2 = axes[1].imshow(log_display.T, 
                     extent=[theta_range[0], theta_range[1], 
                            dev_range[0], dev_range[1]],
                     origin='lower', aspect='auto', cmap=cmap_log,
                     norm=colors.LogNorm(vmin=min_time, vmax=t_span))
axes[1].set_xlabel('Initial Angle (degrees)', fontsize=12)
axes[1].set_ylabel('Deviation Angle (degrees)', fontsize=12)
axes[1].set_title('Double Pendulum Chaos Time (Log Scale, White=Stable)', fontsize=12)
cbar2 = plt.colorbar(im2, ax=axes[1])
cbar2.set_label('Time to Chaos (seconds, log scale)', fontsize=10)

plt.suptitle(f'Double Pendulum Chaos Analysis\nThreshold: {np.rad2deg(CHAOS_THRESHOLD):.2f}°, Max Time: {t_span}s', 
             fontsize=14, fontweight='bold')
plt.tight_layout()

# Save the figure
timestamp = time.strftime("%Y%m%d_%H%M%S")
filename = os.path.join(results_dir, f'double_pendulum_heatmap_{timestamp}.png')
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"✅ Heatmap saved as '{filename}'")

# Also save the raw data
data_filename = os.path.join(results_dir, f'double_pendulum_data_{timestamp}.npz')
np.savez(data_filename,
         chaos_times=chaos_times,
         initial_angles=initial_angles,
         angle_deviations=angle_deviations,
         threshold_rad=CHAOS_THRESHOLD,
         max_time=t_span,
         grid_size=(res_x, res_y))
print(f"✅ Raw data saved as '{data_filename}'")

# --- Print statistics ---
chaotic_points = np.sum(chaos_times < t_span)
stable_points = np.sum(chaos_times >= t_span)
fast_chaotic = np.sum(chaos_times < t_span/2)

print("\n" + "="*70)
print("📊 RESULTS SUMMARY - DOUBLE PENDULUM")
print("="*70)
print(f"Total simulations: {total_sims}")
print(f"Chaotic (<{t_span}s): {chaotic_points} ({chaotic_points/total_sims*100:.1f}%)")
print(f"Fast chaotic (<{t_span/2:.0f}s): {fast_chaotic} ({fast_chaotic/total_sims*100:.1f}%)")
print(f"Stable (={t_span}s): {stable_points} ({stable_points/total_sims*100:.1f}%)")

if len(chaotic_times) > 0:
    print(f"\nChaos time statistics (chaotic region only):")
    print(f"  Minimum: {chaotic_times.min():.1f} seconds")
    print(f"  Maximum: {chaotic_times.max():.1f} seconds")
    print(f"  Mean: {chaotic_times.mean():.1f} seconds")
    print(f"  Median: {np.median(chaotic_times):.1f} seconds")

print("\n" + "="*70)
print("✅ Double Pendulum Analysis Complete!")
print("="*70)

plt.show()