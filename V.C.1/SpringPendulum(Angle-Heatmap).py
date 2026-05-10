# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

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
            norm_up[i] = 0.0001
        if norm_down[i] == 0.0:
            norm_down[i] = 0.0001

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
diffeq_kernel(0.0, dummy_x, 2, np.array([10.0, 10.0]), np.array([1.0, 1.0]),
              np.array([1.0, 1.0]), 0.0, 0.0)


# %%
def simulate_pendulum(theta_initial=np.pi/2, k=[10,10], m=[1,1], L=[1,1], n=2, t_max=1200.0, num_points=700):
    n = 2
    k = np.array(k)
    m = np.array(m)
    L = np.array(L)
    pp = np.array([0.0, 0.0])

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


# %%
def time_to_chaos(theta, dtheta, k=10, threshold=0.5, n=2, t_max=100, num_points=600):
    
    solA = simulate_pendulum(theta_initial=theta, k=[k,k], t_max=t_max, num_points=num_points)
    solB = simulate_pendulum(theta_initial=theta+dtheta, k=[k,k], t_max=t_max, num_points=num_points)

    xA, yA = solA.y[2*(n-1)], solA.y[2*(n-1)+1]
    xB, yB = solB.y[2*(n-1)], solB.y[2*(n-1)+1]

    diff = np.sqrt((xA-xB)**2 + (yA-yB)**2) 
    
    mask = diff > threshold
    if not np.any(mask):
        return solA.t[-1]
    idx = np.argmax(mask)
    return solA.t[idx]


# %%
'''
Fitted Angle vs. Angle Deviation Sim
Double Spring-Pendulum 
N=100
k=10
t_max = 100
num_points = 600
theta range (0, pi/2)
dtheta range (0, 0.25)
threshold (dist between tips) = 0.5
'''

N = 100 
theta_values = np.linspace(0, np.pi/2, N)
dtheta_values = np.linspace(0.0, 0.25, N)

def compute_cell(i, j):
    return time_to_chaos(theta_values[i], dtheta_values[j])

results = Parallel(n_jobs=-1, backend="loky")(
    delayed(compute_cell)(i, j)
    for i in tqdm(range(N), desc="L0 rows")
    for j in range(N)
)

heatmap = np.array(results).reshape(N, N)

plt.figure(figsize=(8, 6))
plt.imshow(heatmap,origin='lower', extent=[dtheta_values[0], dtheta_values[-1], theta_values[0], theta_values[-1]], aspect='auto', cmap='viridis')
plt.colorbar(label='Time to Chaos (s)')
plt.xlabel('Angle Deviation (rad)')
plt.ylabel('Initial Angle')
plt.title('Chaos Divergence Time for Double Spring Pendulums')
plt.tight_layout()
plt.savefig('SpringPen-Heatmap(theta v. dtheta).png')
plt.show()

# %%
'''
Angle vs. Angle Deviation Sim
Double Spring Pendulum
N=100
k=100
t_max = 100
num_points = 600
theta range (0, pi/2)
dtheta range (0, 0.25)
threshold (dist between tips) = 0.5
'''

N = 100 
theta_values = np.linspace(0, np.pi/2, N)
dtheta_values = np.linspace(0.0, 0.25, N)

def compute_cell(i, j):
    return time_to_chaos(theta_values[i], dtheta_values[j], k=100)

results = Parallel(n_jobs=-1, backend="loky")(
    delayed(compute_cell)(i, j)
    for i in tqdm(range(N), desc="L0 rows")
    for j in range(N)
)

heatmap = np.array(results).reshape(N, N)

plt.figure(figsize=(8, 6))
plt.imshow(heatmap,origin='lower', extent=[dtheta_values[0], dtheta_values[-1], theta_values[0], theta_values[-1]], aspect='auto', cmap='viridis')
plt.colorbar(label='Time to Chaos (s)')
plt.xlabel('Angle Deviation (rad)')
plt.ylabel('Initial Angle')
plt.title('Chaos Divergence Time for Double Spring Pendulums')
plt.tight_layout()
plt.savefig('SpringPen-Heatmap(theta v. dtheta).png')
plt.show()

# %%
'''
Angle vs. Angle Deviation Sim
Double Spring Pendulum
N=100
k=1000
t_max = 100
num_points = 600
theta range (0, pi/2)
dtheta range (0, 0.25)
threshold (dist between tips) = 0.5
'''

N = 100 
theta_values = np.linspace(0, np.pi/2, N)
dtheta_values = np.linspace(0.0, 0.25, N)

def compute_cell(i, j):
    return time_to_chaos(theta_values[i], dtheta_values[j], k=1000)

results = Parallel(n_jobs=-1, backend="loky")(
    delayed(compute_cell)(i, j)
    for i in tqdm(range(N), desc="L0 rows")
    for j in range(N)
)

heatmap = np.array(results).reshape(N, N)

plt.figure(figsize=(8, 6))
plt.imshow(heatmap,origin='lower', extent=[dtheta_values[0], dtheta_values[-1], theta_values[0], theta_values[-1]], aspect='auto', cmap='viridis')
plt.colorbar(label='Time to Chaos (s)')
plt.xlabel('Angle Deviation (rad)')
plt.ylabel('Initial Angle')
plt.title('Chaos Divergence Time for Double Spring Pendulums')
plt.tight_layout()
plt.savefig('SpringPen-Heatmap(theta v. dtheta).png')
plt.show()

# %%
'''
Spring Constant vs. Chaos Time graph
'''

dtheta=0.05
plt.figure(figsize=(8, 6))
for theta in np.linspace(0.1, np.pi/2, 2):
    k_array = np.linspace(1,1000,100)
    t = np.array([time_to_chaos(theta, dtheta, k=k, threshold=0.5, n=2, t_max=100, num_points=600) for k in k_array])
    plt.plot(k_array, t, label=f'Angle={theta}')
    
plt.xlabel('Spring Constant (N/m)')
plt.ylabel('Time to Chaos (s)')
plt.title('Chaos Time vs. Spring Constant')
plt.legend()
plt.tight_layout()
plt.savefig('Chaos Time vs. Spring Constant.png')
plt.show()
