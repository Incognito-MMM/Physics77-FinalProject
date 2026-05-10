# %%
# !pip install numba
# %%
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from joblib import Parallel, delayed
from tqdm.notebook import tqdm
from scipy.integrate import solve_ivp

g = 9.8

@njit
def diffeq_kernel(t, x, n, k, L, m, pp0, pp1):
    dxdt = np.zeros_like(x)
    positions = x[:2*n].reshape(n, 2)
    velocities = x[2*n:].reshape(n, 2)

    # Previous points
    prev_points = np.empty_like(positions)
    prev_points[0, 0] = pp0
    prev_points[0, 1] = pp1
    for i in range(1, n):
        prev_points[i, 0] = positions[i-1, 0]
        prev_points[i, 1] = positions[i-1, 1]

    # Next points
    next_points = np.empty_like(positions)
    for i in range(n-1):
        next_points[i, 0] = positions[i+1, 0]
        next_points[i, 1] = positions[i+1, 1]
    next_points[n-1, 0] = positions[n-1, 0]
    next_points[n-1, 1] = positions[n-1, 1]

    # Vectors
    vec_up = positions - prev_points
    vec_down = next_points - positions

    # Norms
    norm_up = np.sqrt(vec_up[:, 0]**2 + vec_up[:, 1]**2)
    norm_down = np.sqrt(vec_down[:, 0]**2 + vec_down[:, 1]**2)

    # Avoid division by zero
    for i in range(n):
        if norm_up[i] == 0.0:
            norm_up[i] = 1.0
        if norm_down[i] == 0.0:
            norm_down[i] = 1.0

    # Forces from springs
    F_up = np.empty_like(positions)
    F_down = np.empty_like(positions)
    for i in range(n):
        stretch_up = norm_up[i] - L[i]
        F_up[i, 0] = -k[i] * stretch_up * vec_up[i, 0] / norm_up[i]
        F_up[i, 1] = -k[i] * stretch_up * vec_up[i, 1] / norm_up[i]

        stretch_down = norm_down[i] - L[i]
        F_down[i, 0] =  k[i] * stretch_down * vec_down[i, 0] / norm_down[i]
        F_down[i, 1] =  k[i] * stretch_down * vec_down[i, 1] / norm_down[i]

    # Gravity
    gravity = np.empty_like(positions)
    for i in range(n):
        gravity[i, 0] = 0.0
        gravity[i, 1] = -m[i] * g

    # Accelerations
    acc = np.empty_like(positions)
    for i in range(n):
        acc[i, 0] = (F_up[i, 0] + F_down[i, 0] + gravity[i, 0]) / m[i]
        acc[i, 1] = (F_up[i, 1] + F_down[i, 1] + gravity[i, 1]) / m[i]

    dxdt[:2*n] = velocities.reshape(2*n)
    dxdt[2*n:] = acc.reshape(2*n)
    return dxdt

def make_rhs(n, k, L, m, pp):
    k = np.asarray(k)
    L = np.asarray(L)
    m = np.asarray(m)
    pp0, pp1 = pp

    def rhs(t, x):
        return diffeq_kernel(t, x, n, k, L, m, pp0, pp1)
    return rhs

# Pre-JIT compile with a dummy call
dummy_x = np.zeros(8)  # for n=2 → 4 positions + 4 velocities
diffeq_kernel(0.0, dummy_x, 2, np.array([10.0, 10.0]), np.array([1.0, 1.0]), np.array([1.0, 1.0]), 0.0, 0.0)


# %%
def simulate_pendulum(theta_initial=np.pi/2, k=[10,10], m=[1,1], L=[1,1], n=2, t_max=1200.0, num_points=700):
    n = 2
    k = np.array(k)
    m = np.array(m)
    L = np.array(L)
    pp = np.array([0.0, 0.0])
    theta_initial = np.array(theta_initial)

    x0 = np.array([
        L[0] * np.sin(theta_initial), -L[0] * np.cos(theta_initial),   # mass 1 position
        (L[0] + L[1]) * np.sin(theta_initial), -(L[0] + L[1]) * np.cos(theta_initial),   # mass 2 position
        0.0, 0.0,      # mass 1 velocity
        0.0, 0.0       # mass 2 velocity
    ])

    t_eval = np.linspace(0.0, t_max, num_points)
    rhs = make_rhs(n, k, L, m, pp)
    sol = solve_ivp(rhs, [0.0, t_max], x0, t_eval=t_eval, method="DOP853", rtol=1e-5, atol=1e-7)
    
    return sol

# time to chaos based on angle deviation at tip
def time_to_chaos_angle(L_base, dL, threshold=0.5, n=2, t_max=100, num_points=600):
    
    # Two systems with slightly different spring lengths
    solA = simulate_pendulum(np.pi/3, [L_base, L_base], t_max=t_max, num_points=num_points)
    solB = simulate_pendulum(np.pi/3, [L_base + dL, L_base + dL], t_max=t_max, num_points=num_points)

    # Positions of the two masses
    x1A, y1A = solA.y[0], solA.y[1]
    x2A, y2A = solA.y[2], solA.y[3]

    x1B, y1B = solB.y[0], solB.y[1]
    x2B, y2B = solB.y[2], solB.y[3]

    # Angles of second mass relative to first (vector from mass1 to mass2)
    dxA = x2A - x1A
    dyA = y2A - y1A
    dxB = x2B - x1B
    dyB = y2B - y1B

    thetaA = np.arctan2(dyA, dxA)
    thetaB = np.arctan2(dyB, dxB)

    diff = np.abs(thetaA - thetaB)
    diff = np.mod(diff + np.pi, 2*np.pi) - np.pi
    diff = np.abs(diff)

    mask = diff > threshold
    if not np.any(mask):
        return solA.t[-1]
    idx = np.argmax(mask)
    return solA.t[idx]

def time_to_chaos_distance(L_base, dL, threshold=0.5, n=2, t_max=100, num_points=600):
    # Two systems with slightly different spring lengths
    solA = simulate_pendulum(theta_initial=np.pi/6, L=[L_base, L_base], t_max=t_max, num_points=num_points)
    solB = simulate_pendulum(theta_initial=np.pi/6, L=[L_base + dL, L_base + dL], t_max=t_max, num_points=num_points)

    # Positions of the two masses at the tip
    xA, yA = solA.y[2], solA.y[3]
    xB, yB = solB.y[2], solB.y[3]

    diff = np.sqrt((xA-xB)**2 + (yA-yB)**2) 

    mask = diff > threshold
    if not np.any(mask):
        return solA.t[-1]
    idx = np.argmax(mask)
    return solA.t[idx]


# %%
'''
Equlibreum Length vs. Equilibreum Deviation Sim
Double Spring Pendulum
N=100
k=10
t_max = 100
num_points = 600
L (equilibreum length) range (0.2, 4)
dL range (0, 0.25)
threshold (dist between tips) = 0.5
'''

N = 100 
L_values = np.linspace(0.2, 4, N)
dL_values = np.linspace(0.0, 0.1, N)

def compute_cell(i, j):
    return time_to_chaos_distance(L_values[i], dL_values[j])

results = Parallel(n_jobs=-1, backend="loky")(
    delayed(compute_cell)(i, j)
    for i in tqdm(range(N), desc="L0 rows")
    for j in range(N)
)

heatmap = np.array(results).reshape(N, N)

plt.figure(figsize=(8, 6))
plt.imshow(
    heatmap,
    origin='lower',
    extent=[dL_values[0], dL_values[-1], L_values[0], L_values[-1]],
    aspect='auto',
    cmap='inferno'
)
plt.colorbar(label='Time to Chaos (s)')
plt.xlabel('Initial ΔL (m)')
plt.ylabel('Base Length L₀ (m)')
plt.title('Chaos Divergence Time for Double Spring Pendulums')
plt.tight_layout()
plt.savefig('SpringPen-Heatmap.png')
plt.show()
