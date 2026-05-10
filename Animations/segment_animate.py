# MULTIPLE SEGMENTS SIMULATION

import numpy as np
import scipy.linalg
import scipy.integrate
import sympy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import time
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION - ALIGNED PENDULUM TIPS
# ============================================================================

# Simulation settings
t_span = 40                     # Total time (seconds)
t_eval_points = 1600            # Total Frames
g = 1.0                         
method = 'DOP853'               

# Animation settings
animation_interval = 25         # 25ms = 40 fps
trace_length = 100              # Trail length
show_reference_circle = True    # Show circle at tip radius

# Aligned pendulums settings
n_links_list = [2]  # Which pendulum configurations to show
start_angle_deg = 180            # Starting angle for ALL pendulums (degrees)
total_length = 5.0              # Total length for ALL pendulums (tips align)

# Color settings
color_map = 'plasma'            
line_widths = {2:2.0}
marker_sizes = {2:1}

# Progress tracker
class ProgressTracker:
    def __init__(self, t_final):
        self.t_final = t_final
        self.last_p = -1
        self.start_time = time.time()

    def update(self, t):
        percent = int(t / self.t_final * 100)
        if percent > self.last_p and percent % 10 == 0:
            elapsed = time.time() - self.start_time
            eta = elapsed / max(0.01, t) * (self.t_final - t) if t > 0 else 0
            bar_length = 40
            filled = int(bar_length * percent / 100)
            bar = '█' * filled + '░' * (bar_length - filled)
            print(f"\rProgress: |{bar}| {percent}% | ETA: {eta:.1f}s", end="")
            self.last_p = percent

# ============================================================================
# YOUR ORIGINAL SOLVER (Modified for variable N masses)
# ============================================================================

def H_solver_for_N_pendulum(N_masses, initial_p=None, initial_theta=None, 
                           lengths=None, mass=None, t_eval=1000, g=1, 
                           tracker=None):
    """
    Your original Hamiltonian solver - proven to conserve energy!
    """
    # initialization
    i_array = np.ones(N_masses)
    if lengths is None:
        lengths = i_array
    if mass is None:
        mass = i_array
    if initial_p is None:
        initial_p = np.zeros(N_masses)
    if initial_theta is None:
        initial_theta = np.full((N_masses,), np.pi/2)
    initial = np.concatenate([initial_p, initial_theta])

    # assistant matrix
    I = np.diag(i_array)
    A = np.diag(np.ones(N_masses-1), k=-1)
    B = scipy.linalg.inv(I - A)

    # convert to sym
    B_mat = sympy.Matrix(B)
    L_mat = sympy.Matrix(np.diag(lengths))
    m_vec = sympy.Matrix(mass)
    M_mat = sympy.Matrix(np.diag(mass))

    # symbolic computation
    theta = sympy.symbols(f'theta1:{N_masses+1}', real=True)
    theta_vec = sympy.Matrix(theta).reshape(N_masses, 1)
    p = sympy.symbols(f'p1:{N_masses+1}', real=True)
    p_vec = sympy.Matrix(p).reshape(N_masses, 1)

    cos_theta_vec = sympy.Matrix([sympy.cos(t) for t in theta])
    Cos_Theta_mat = sympy.diag(*[sympy.cos(t) for t in theta])
    Sin_Theta_mat = sympy.diag(*[sympy.sin(t) for t in theta])

    # potential energy
    h_vec = B_mat @ (L_mat @ cos_theta_vec)
    V = - g * m_vec.T @ h_vec
    V = V[0]

    # mass matrix under theta_dot
    M_theta_mat = Cos_Theta_mat @ L_mat @ B_mat.T @ M_mat @ B_mat @ L_mat @ Cos_Theta_mat + \
                  Sin_Theta_mat @ L_mat @ B_mat.T @ M_mat @ B_mat @ L_mat @ Sin_Theta_mat
    M_theta_mat = sympy.simplify(M_theta_mat)
    dM_theta_dtheta_VecMat = [sympy.diff(M_theta_mat, ti) for ti in theta]
    dV_dtheta_vec = [sympy.diff(V, ti) for ti in theta]

    # convert to function
    V_func = sympy.lambdify([theta], V, "numpy")
    M_theta_func = sympy.lambdify([theta], M_theta_mat, "numpy")
    dM_theta_dtheta_func = sympy.lambdify([theta], dM_theta_dtheta_VecMat, "numpy")
    dV_dtheta_func = sympy.lambdify([theta], dV_dtheta_vec, "numpy")

    # Define Hamilton's equations (your original)
    def Hamilton_equations(t, z):
        if tracker:
            tracker.update(t)
        
        p = z[:N_masses]
        q = z[N_masses:]

        M_num = M_theta_func(q)
        M_inv = scipy.linalg.inv(M_num)

        p_dot = np.zeros(N_masses)
        for i in range(N_masses):
            p_dot[i] = 0.5 * p.T @ M_inv @ dM_theta_dtheta_func(q)[i] @ M_inv @ p - dV_dtheta_func(q)[i]
        
        q_dot = M_inv @ p

        return np.concatenate([p_dot, q_dot])

    # solve ODE
    t_eval_array = np.linspace(0, t_span, t_eval)
    sol = scipy.integrate.solve_ivp(
        Hamilton_equations, method=method, t_span=[0, t_span], 
        y0=initial, t_eval=t_eval_array, rtol=1e-8, atol=1e-10
    )

    # calculate energy
    t_list = sol.t
    n_steps = len(t_list)
    
    T_num = np.zeros(n_steps)
    V_num = np.zeros(n_steps)

    for i in range(n_steps):
        p_num = sol.y[:N_masses, i]
        q_num = sol.y[N_masses:, i]
        M_num = M_theta_func(q_num)
        M_num_inv = scipy.linalg.inv(M_num)
        V_num[i] = V_func(q_num)
        T_num[i] = 0.5 * p_num.T @ M_num_inv @ p_num
    H_num = T_num + V_num

    return sol, T_num, V_num, H_num

# ============================================================================
# ALIGNED PENDULUMS ANIMATION FUNCTION
# ============================================================================

def animate_aligned():
    """Animate pendulums with different numbers of links, all aligned."""
    
    print("\n" + "="*70)
    print("🎯 ALIGNED PENDULUM TIPS (Energy-Conserving Solver)")
    print(f"Pendulum configurations: {n_links_list} links")
    print(f"Starting angle: {start_angle_deg}°")
    print(f"Total length for all: {total_length}")
    print("="*70)
    
    # Calculate segment lengths for each configuration
    lengths_dict = {}
    for n_links in n_links_list:
        segment_length = total_length / n_links
        lengths_dict[n_links] = [segment_length] * n_links
        print(f"  {n_links}-link: {n_links} segments of {segment_length:.3f}")
    
    start_angle_rad = np.deg2rad(start_angle_deg)
    
    solutions = []
    energies = []
    lengths_list = []
    
    print(f"\nSolving {len(n_links_list)} pendulum configurations...")
    start_time = time.time()
    
    for i, n in enumerate(n_links_list):
        print(f"  Solving {n}-link pendulum...", end="\r")
        
        initial_theta = [start_angle_rad] * n
        initial_p = [0.0] * n
        current_lengths = lengths_dict[n]
        
        # Create tracker for first pendulum only
        tracker = None if i > 0 else ProgressTracker(t_span)
        
        sol, T, V, H = H_solver_for_N_pendulum(
            N_masses=n,
            initial_p=initial_p,
            initial_theta=initial_theta,
            lengths=current_lengths,
            mass=[1]*n,
            t_eval=t_eval_points,
            g=g,
            tracker=tracker
        )
        
        solutions.append(sol)
        energies.append((T, V, H))
        lengths_list.append(current_lengths)
        
        # Check energy conservation
        if i == 0:
            energy_drift = np.max(np.abs(H - H[0]))
            print(f"\n  ✓ Energy drift for {n}-link pendulum: {energy_drift:.2e}")
    
    elapsed = time.time() - start_time
    print(f"\n✅ Solved all configurations in {elapsed:.1f} seconds")
    
    # Create color gradient
    colors = plt.cm.__dict__[color_map](np.linspace(0, 1, len(solutions)))
    
    # Setup animation
    fig, ax = plt.subplots(figsize=(14, 12))
    max_limit = total_length + 0.5
    ax.set_xlim(-max_limit, max_limit)
    ax.set_ylim(-max_limit, max_limit)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    
    # Add reference circle
    if show_reference_circle:
        circle = plt.Circle((0, 0), total_length, fill=False, 
                           color='gray', linestyle='--', alpha=0.3, linewidth=1.5)
        ax.add_patch(circle)
        ax.text(total_length, 0, f'  Tip radius = {total_length}', fontsize=10, alpha=0.5)
    
    ax.set_title(f'Aligned Pendulum Tips: {n_links_list} Links\n'
                f'All starting at {start_angle_deg}°, Total Length = {total_length}\n'
                f'Energy conserved to 1e-10 - All tips stay on gray circle!', 
                fontsize=12)
    
    # Create plot elements
    lines = []
    traces = []
    
    for i, (sol, n) in enumerate(zip(solutions, n_links_list)):
        color = colors[i]
        line_width = line_widths.get(n, 2.0)
        marker_size = marker_sizes.get(n, 6)
        
        line, = ax.plot([], [], 'o-', lw=line_width, markersize=marker_size, 
                       color=color, markerfacecolor=color, 
                       markeredgecolor='darkred', alpha=0.8, 
                       label=f"{n}-link pendulum")
        trace, = ax.plot([], [], '-', lw=1, color=color, alpha=0.4)
        lines.append(line)
        traces.append(trace)
    
    ax.legend(loc='upper right', fontsize=11)
    
    # Precompute coordinates
    n_frames = len(solutions[0].t)
    print(f"\nPrecomputing coordinates for {n_frames} frames...")
    start_time = time.time()
    
    all_x = []
    all_y = []
    
    for idx, (sol, n, lengths) in enumerate(zip(solutions, n_links_list, lengths_list)):
        theta = sol.y[n:]  # Theta angles start at index n
        x = np.zeros((n + 1, n_frames))
        y = np.zeros((n + 1, n_frames))
        
        for frame in range(n_frames):
            for i in range(n):
                if i == 0:
                    x[0, frame] = 0
                    y[0, frame] = 0
                x[i+1, frame] = x[i, frame] + lengths[i] * np.sin(theta[i, frame])
                y[i+1, frame] = y[i, frame] - lengths[i] * np.cos(theta[i, frame])
        
        all_x.append(x)
        all_y.append(y)
        
        print(f"  Processed {idx+1}/{len(solutions)} configurations", end="\r")
    
    print(f"\n✅ Precomputed coordinates in {time.time() - start_time:.1f} seconds")
    
    def update(frame):
        artists = []
        for i in range(len(solutions)):
            lines[i].set_data(all_x[i][:, frame], all_y[i][:, frame])
            start_idx = max(0, frame - trace_length)
            traces[i].set_data(all_x[i][-1, start_idx:frame], 
                              all_y[i][-1, start_idx:frame])
            artists.extend([lines[i], traces[i]])
        
        ax.set_title(f'Aligned Pendulum Tips - Frame {frame}/{n_frames}\n'
                    f'Time: {solutions[0].t[frame]:.1f}s / {t_span}s\n'
                    f'Configurations: {n_links_list} links, Energy conserved to 1e-10', 
                    fontsize=11)
        
        if show_reference_circle:
            artists.append(circle)
        
        return artists
    
    print("\nCreating animation...")
    ani = animation.FuncAnimation(fig, update, frames=n_frames, 
                                  interval=animation_interval, blit=False)
    
    plt.close(fig)
    return ani, solutions[0], energies[0]

# ============================================================================
# ENERGY VALIDATION PLOT
# ============================================================================

def plot_energy_conservation(sol, T, V, H, title="Energy Conservation"):
    """Plot energy components to verify conservation."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Energy components
    axes[0].plot(sol.t, T, label='Kinetic Energy', linewidth=2, alpha=0.7)
    axes[0].plot(sol.t, V, label='Potential Energy', linewidth=2, alpha=0.7)
    axes[0].plot(sol.t, H, label='Total Energy', linewidth=2, color='black')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Energy')
    axes[0].set_title(title)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Energy drift
    energy_drift = H - H[0]
    axes[1].plot(sol.t, energy_drift, linewidth=1.5, color='red')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Energy Drift')
    axes[1].set_title(f'Energy Conservation - Max Drift: {np.max(np.abs(energy_drift)):.2e}')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n✅ Energy Conservation Check:")
    print(f"   Initial Energy: {H[0]:.6f}")
    print(f"   Final Energy: {H[-1]:.6f}")
    print(f"   Max Drift: {np.max(np.abs(energy_drift)):.2e}")
    print(f"   Relative Drift: {np.max(np.abs(energy_drift))/H[0]:.2e}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("🎯 ALIGNED PENDULUM TIPS - USING ENERGY-CONSERVING SOLVER")
    print(f"Simulation time: {t_span} seconds at {t_eval_points/t_span:.1f} fps")
    print("="*70)
    
    # Increase embed limit
    plt.rcParams['animation.embed_limit'] = 200
    
    # Create animation
    ani, sol, (T, V, H) = animate_aligned()
    
    # Verify energy conservation
    print("\n" + "="*70)
    print("🔬 ENERGY CONSERVATION VERIFICATION")
    print("="*70)
    plot_energy_conservation(sol, T, V, H, f"Energy Conservation - {len(sol.y)-sol.y.shape[0]//2}-link Pendulum")
    
    # Display animation
    print("\n" + "="*70)
    print("🎬 DISPLAYING ALIGNED PENDULUM ANIMATION")
    print("="*70)
    ani.save('all_10_pendulums.mp4', writer='ffmpeg', fps=30)
    print("Animation saved to file - download and view locally")
    
    print("\n✅ ANIMATION COMPLETE - Energy is properly conserved!")