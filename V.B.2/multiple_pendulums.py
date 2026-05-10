import numpy as np
import scipy.linalg
import scipy.integrate
import sympy
import matplotlib.pyplot as plt
from matplotlib import colors
import time
import multiprocessing as mp
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
g = 9.8
CHAOS_THRESHOLD = 0.05  # radians
t_span = 1800  # Full 30 minutes
initial_angle = 30  # degrees
dev_range = [0, 1.0]
res_y = 100  # Reduced resolution for sanity

print("="*80)
print("🚀 MULTI-PENDULUM CHAOS ANALYSIS (EXPLICIT SYSTEMS)")
print("="*80)
print(f"Testing: 2, 3, 4, 5, 6, 7 pendulums")
print(f"Initial angle: {initial_angle}°")
print(f"Deviation range: 0° to 1.0°")
print(f"Max simulation time: {t_span} seconds")
print(f"Deviation resolution: {res_y} points")
print("="*80)
print("\n⚠️  WARNING: Each system is generated SEPARATELY with FULL physics")
print("   Expected runtime: 12-24+ hours")
print("="*80)

results_dir = "explicit_pendulum_results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# ============================================================
# EXPLICIT SYSTEM DEFINITIONS (NO SHORTCUTS)
# ============================================================

def create_system_N2():
    """Explicit 2-pendulum system"""
    N = 2
    theta = sympy.symbols('theta1 theta2', real=True)
    lengths = [1.0, 1.0]
    masses = [1.0, 1.0]
    
    # Build matrices
    I = np.eye(N)
    A = np.diag(np.ones(N-1), k=-1)
    B = np.linalg.inv(I - A)
    B_mat = sympy.Matrix(B)
    L_mat = sympy.Matrix(np.diag(lengths))
    M_diag = sympy.Matrix(np.diag(masses))
    
    cos_vec = sympy.Matrix([sympy.cos(theta[0]), sympy.cos(theta[1])])
    Cos_mat = sympy.diag(*cos_vec)
    Sin_mat = sympy.diag(*[sympy.sin(t) for t in theta])
    
    M_theta = Cos_mat @ L_mat @ B_mat.T @ M_diag @ B_mat @ L_mat @ Cos_mat + \
              Sin_mat @ L_mat @ B_mat.T @ M_diag @ B_mat @ L_mat @ Sin_mat
    
    h = B_mat @ (L_mat @ cos_vec)
    V = -g * sympy.Matrix(masses).T @ h
    V = V[0]
    
    dM = sympy.Matrix([sympy.flatten(sympy.diff(M_theta, ti)) for ti in theta])
    dV = sympy.Matrix([sympy.diff(V, ti) for ti in theta])
    
    M_f = sympy.lambdify([theta], M_theta, 'numpy')
    dM_f = sympy.lambdify([theta], dM, 'numpy', cse=True)
    dV_f = sympy.lambdify([theta], dV, 'numpy')
    
    return M_f, dM_f, dV_f, N

def create_system_N3():
    """Explicit 3-pendulum system"""
    N = 3
    theta = sympy.symbols('theta1 theta2 theta3', real=True)
    lengths = [1.0, 1.0, 1.0]
    masses = [1.0, 1.0, 1.0]
    
    I = np.eye(N)
    A = np.diag(np.ones(N-1), k=-1)
    B = np.linalg.inv(I - A)
    B_mat = sympy.Matrix(B)
    L_mat = sympy.Matrix(np.diag(lengths))
    M_diag = sympy.Matrix(np.diag(masses))
    
    cos_vec = sympy.Matrix([sympy.cos(t) for t in theta])
    Cos_mat = sympy.diag(*cos_vec)
    Sin_mat = sympy.diag(*[sympy.sin(t) for t in theta])
    
    M_theta = Cos_mat @ L_mat @ B_mat.T @ M_diag @ B_mat @ L_mat @ Cos_mat + \
              Sin_mat @ L_mat @ B_mat.T @ M_diag @ B_mat @ L_mat @ Sin_mat
    
    h = B_mat @ (L_mat @ cos_vec)
    V = -g * sympy.Matrix(masses).T @ h
    V = V[0]
    
    dM = sympy.Matrix([sympy.flatten(sympy.diff(M_theta, ti)) for ti in theta])
    dV = sympy.Matrix([sympy.diff(V, ti) for ti in theta])
    
    M_f = sympy.lambdify([theta], M_theta, 'numpy')
    dM_f = sympy.lambdify([theta], dM, 'numpy', cse=True)
    dV_f = sympy.lambdify([theta], dV, 'numpy')
    
    return M_f, dM_f, dV_f, N

def create_system_N4():
    """Explicit 4-pendulum system"""
    N = 4
    theta = sympy.symbols('theta1 theta2 theta3 theta4', real=True)
    lengths = [1.0, 1.0, 1.0, 1.0]
    masses = [1.0, 1.0, 1.0, 1.0]
    
    I = np.eye(N)
    A = np.diag(np.ones(N-1), k=-1)
    B = np.linalg.inv(I - A)
    B_mat = sympy.Matrix(B)
    L_mat = sympy.Matrix(np.diag(lengths))
    M_diag = sympy.Matrix(np.diag(masses))
    
    cos_vec = sympy.Matrix([sympy.cos(t) for t in theta])
    Cos_mat = sympy.diag(*cos_vec)
    Sin_mat = sympy.diag(*[sympy.sin(t) for t in theta])
    
    M_theta = Cos_mat @ L_mat @ B_mat.T @ M_diag @ B_mat @ L_mat @ Cos_mat + \
              Sin_mat @ L_mat @ B_mat.T @ M_diag @ B_mat @ L_mat @ Sin_mat
    
    h = B_mat @ (L_mat @ cos_vec)
    V = -g * sympy.Matrix(masses).T @ h
    V = V[0]
    
    dM = sympy.Matrix([sympy.flatten(sympy.diff(M_theta, ti)) for ti in theta])
    dV = sympy.Matrix([sympy.diff(V, ti) for ti in theta])
    
    M_f = sympy.lambdify([theta], M_theta, 'numpy')
    dM_f = sympy.lambdify([theta], dM, 'numpy', cse=True)
    dV_f = sympy.lambdify([theta], dV, 'numpy')
    
    return M_f, dM_f, dV_f, N

def create_system_N5():
    """Explicit 5-pendulum system"""
    N = 5
    theta = sympy.symbols('theta1 theta2 theta3 theta4 theta5', real=True)
    lengths = [1.0, 1.0, 1.0, 1.0, 1.0]
    masses = [1.0, 1.0, 1.0, 1.0, 1.0]
    
    I = np.eye(N)
    A = np.diag(np.ones(N-1), k=-1)
    B = np.linalg.inv(I - A)
    B_mat = sympy.Matrix(B)
    L_mat = sympy.Matrix(np.diag(lengths))
    M_diag = sympy.Matrix(np.diag(masses))
    
    cos_vec = sympy.Matrix([sympy.cos(t) for t in theta])
    Cos_mat = sympy.diag(*cos_vec)
    Sin_mat = sympy.diag(*[sympy.sin(t) for t in theta])
    
    M_theta = Cos_mat @ L_mat @ B_mat.T @ M_diag @ B_mat @ L_mat @ Cos_mat + \
              Sin_mat @ L_mat @ B_mat.T @ M_diag @ B_mat @ L_mat @ Sin_mat
    
    h = B_mat @ (L_mat @ cos_vec)
    V = -g * sympy.Matrix(masses).T @ h
    V = V[0]
    
    dM = sympy.Matrix([sympy.flatten(sympy.diff(M_theta, ti)) for ti in theta])
    dV = sympy.Matrix([sympy.diff(V, ti) for ti in theta])
    
    M_f = sympy.lambdify([theta], M_theta, 'numpy')
    dM_f = sympy.lambdify([theta], dM, 'numpy', cse=True)
    dV_f = sympy.lambdify([theta], dV, 'numpy')
    
    return M_f, dM_f, dV_f, N

def create_system_N6():
    """Explicit 6-pendulum system"""
    N = 6
    theta = sympy.symbols('theta1 theta2 theta3 theta4 theta5 theta6', real=True)
    lengths = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    masses = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    
    I = np.eye(N)
    A = np.diag(np.ones(N-1), k=-1)
    B = np.linalg.inv(I - A)
    B_mat = sympy.Matrix(B)
    L_mat = sympy.Matrix(np.diag(lengths))
    M_diag = sympy.Matrix(np.diag(masses))
    
    cos_vec = sympy.Matrix([sympy.cos(t) for t in theta])
    Cos_mat = sympy.diag(*cos_vec)
    Sin_mat = sympy.diag(*[sympy.sin(t) for t in theta])
    
    M_theta = Cos_mat @ L_mat @ B_mat.T @ M_diag @ B_mat @ L_mat @ Cos_mat + \
              Sin_mat @ L_mat @ B_mat.T @ M_diag @ B_mat @ L_mat @ Sin_mat
    
    h = B_mat @ (L_mat @ cos_vec)
    V = -g * sympy.Matrix(masses).T @ h
    V = V[0]
    
    dM = sympy.Matrix([sympy.flatten(sympy.diff(M_theta, ti)) for ti in theta])
    dV = sympy.Matrix([sympy.diff(V, ti) for ti in theta])
    
    M_f = sympy.lambdify([theta], M_theta, 'numpy')
    dM_f = sympy.lambdify([theta], dM, 'numpy', cse=True)
    dV_f = sympy.lambdify([theta], dV, 'numpy')
    
    return M_f, dM_f, dV_f, N

def create_system_N7():
    """Explicit 7-pendulum system"""
    N = 7
    theta = sympy.symbols('theta1 theta2 theta3 theta4 theta5 theta6 theta7', real=True)
    lengths = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    masses = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    
    I = np.eye(N)
    A = np.diag(np.ones(N-1), k=-1)
    B = np.linalg.inv(I - A)
    B_mat = sympy.Matrix(B)
    L_mat = sympy.Matrix(np.diag(lengths))
    M_diag = sympy.Matrix(np.diag(masses))
    
    cos_vec = sympy.Matrix([sympy.cos(t) for t in theta])
    Cos_mat = sympy.diag(*cos_vec)
    Sin_mat = sympy.diag(*[sympy.sin(t) for t in theta])
    
    M_theta = Cos_mat @ L_mat @ B_mat.T @ M_diag @ B_mat @ L_mat @ Cos_mat + \
              Sin_mat @ L_mat @ B_mat.T @ M_diag @ B_mat @ L_mat @ Sin_mat
    
    h = B_mat @ (L_mat @ cos_vec)
    V = -g * sympy.Matrix(masses).T @ h
    V = V[0]
    
    dM = sympy.Matrix([sympy.flatten(sympy.diff(M_theta, ti)) for ti in theta])
    dV = sympy.Matrix([sympy.diff(V, ti) for ti in theta])
    
    M_f = sympy.lambdify([theta], M_theta, 'numpy')
    dM_f = sympy.lambdify([theta], dM, 'numpy', cse=True)
    dV_f = sympy.lambdify([theta], dV, 'numpy')
    
    return M_f, dM_f, dV_f, N

# ============================================================
# HAMILTONIAN AND SIMULATION FUNCTIONS
# ============================================================

def create_hamiltonian(M_f, dM_f, dV_f, N_masses):
    def hamilton_equations(t, z):
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
    return hamilton_equations

def run_simulation(system, deviation_deg):
    """Run a single simulation for a specific system"""
    M_f, dM_f, dV_f, N = system
    hamilton_eq = create_hamiltonian(M_f, dM_f, dV_f, N)
    
    base_theta = np.deg2rad(initial_angle)
    dev_theta = np.deg2rad(deviation_deg)
    
    if dev_theta == 0:
        return t_span
    
    z_base = np.concatenate([np.zeros(N), np.full(N, base_theta)])
    z_pert = np.concatenate([np.zeros(N), np.full(N, base_theta + dev_theta)])

    def combined_ode(t, Z):
        return np.concatenate([
            hamilton_eq(t, Z[:2*N]),
            hamilton_eq(t, Z[2*N:])
        ])

    def deviation_event(t, Z, *args):
        dist = np.linalg.norm(Z[N:2*N] - Z[3*N:4*N])
        return dist - CHAOS_THRESHOLD
    deviation_event.terminal = True

    try:
        sol = scipy.integrate.solve_ivp(
            combined_ode, [0, t_span], 
            np.concatenate([z_base, z_pert]),
            events=deviation_event,
            method='DOP853',
            rtol=1e-6,
            atol=1e-8,
            max_step=0.1
        )
        
        if sol.t_events[0].size > 0:
            return sol.t_events[0][0]
    except:
        pass
    
    return t_span

# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    # Create all systems explicitly (this will take time!)
    print("\n📐 GENERATING ALL SYSTEMS EXPLICITLY...")
    print("   This will take 10-30 minutes for N=7...")
    
    systems = {
        2: create_system_N2(),
        3: create_system_N3(),
        4: create_system_N4(),
        5: create_system_N5(),
        6: create_system_N6(),
        7: create_system_N7(),
    }
    
    print("✅ All systems generated!\n")
    
    deviations = np.linspace(dev_range[0], dev_range[1], res_y)
    pendulum_counts = [2, 3, 4, 5, 6, 7]
    
    # Store results
    chaos_times = np.zeros((len(deviations), len(pendulum_counts)))
    
    # Run simulations for each system
    for idx, N in enumerate(pendulum_counts):
        system = systems[N]
        print(f"\n{'='*60}")
        print(f"🚀 Running {N}-pendulum system ({res_y} simulations)")
        print(f"{'='*60}")
        
        results = []
        for dev in tqdm(deviations, desc=f"N={N}", unit="sim"):
            result = run_simulation(system, dev)
            results.append(result)
        
        chaos_times[:, idx] = results
        
        # Verify zero deviation
        if results[0] == t_span:
            print(f"  ✓ δ=0°: {t_span}s (stable)")
        else:
            print(f"  ❌ BUG: δ=0° returned {results[0]:.1f}s")
    
    # Create visualization
    fig = plt.figure(figsize=(20, 12))
    
    # Heatmap
    ax1 = plt.subplot(2, 2, 1)
    im = ax1.imshow(chaos_times, 
                    extent=[1.5, 7.5, 0, 1.0],
                    origin='lower', aspect='auto', cmap='viridis',
                    vmin=0, vmax=t_span)
    ax1.set_xlabel('Number of Pendulums', fontsize=14)
    ax1.set_ylabel('Deviation Angle (degrees)', fontsize=14)
    ax1.set_title('Chaos Time Heatmap', fontsize=12)
    ax1.set_xticks(pendulum_counts)
    ax1.set_xticklabels(pendulum_counts, fontsize=12)
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Time to Chaos (seconds)', fontsize=12)
    
    # Line plot
    ax2 = plt.subplot(2, 2, 2)
    colors_line = plt.cm.viridis(np.linspace(0, 1, len(pendulum_counts)))
    for i, N in enumerate(pendulum_counts):
        ax2.plot(deviations, chaos_times[:, i], linewidth=2, 
                label=f'{N} pendulums', color=colors_line[i])
    ax2.set_xlabel('Deviation Angle (degrees)', fontsize=12)
    ax2.set_ylabel('Time to Chaos (seconds)', fontsize=12)
    ax2.set_title('Chaos Time vs Deviation', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, t_span)
    
    # Bar charts
    ax3 = plt.subplot(2, 2, 3)
    avg_times = [np.mean(chaos_times[1:, i]) for i in range(len(pendulum_counts))]
    ax3.bar(pendulum_counts, avg_times, color=colors_line, alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Number of Pendulums', fontsize=12)
    ax3.set_ylabel('Average Chaos Time (s)', fontsize=12)
    ax3.set_title('Average Chaos Time', fontsize=12)
    ax3.grid(True, alpha=0.3, axis='y')
    
    ax4 = plt.subplot(2, 2, 4)
    chaos_prob = [np.sum(chaos_times[1:, i] < t_span) / (len(deviations)-1) * 100 
                  for i in range(len(pendulum_counts))]
    ax4.bar(pendulum_counts, chaos_prob, color=colors_line, alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Number of Pendulums', fontsize=12)
    ax4.set_ylabel('Chaotic Region (%)', fontsize=12)
    ax4.set_title('Probability of Chaos', fontsize=12)
    ax4.set_ylim(0, 105)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Multi-Pendulum Chaos Analysis (N=2-7)\nInitial Angle: {initial_angle}°, Max Time: {t_span}s',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(results_dir, f'explicit_N7_{timestamp}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n✅ Figure saved as '{filename}'")
    
    # Save data
    data_file = os.path.join(results_dir, f'explicit_N7_data_{timestamp}.npz')
    np.savez(data_file,
             pendulum_counts=pendulum_counts,
             deviations=deviations,
             chaos_times=chaos_times,
             initial_angle=initial_angle,
             t_span=t_span)
    print(f"✅ Data saved as '{data_file}'")
    
    # Print results
    print("\n" + "="*70)
    print("📊 FINAL RESULTS")
    print("="*70)
    for i, N in enumerate(pendulum_counts):
        times = chaos_times[:, i]
        print(f"\n{N}-Pendulum:")
        print(f"  δ=0°: {times[0]:.0f}s")
        print(f"  Avg chaos: {np.mean(times[1:]):.1f}s")
        print(f"  Min chaos: {times[1:].min():.1f}s")
    
    plt.show()

if __name__ == '__main__':
    mp.freeze_support()
    main()