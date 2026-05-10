import numpy as np
import pandas as pd
from scipy.special import jn_zeros, j1
from scipy.integrate import cumulative_trapezoid
from tqdm import tqdm

def calculate_theoretical_asymptote(L, g, theta_deg, duration, dt, n_modes=200):
    """
    Calculates the continuum limit path length for a rope using Bessel modes
    with a geometric correction for vertical velocity components.
    """
    t = np.arange(0, duration, dt)
    theta0_rad = np.radians(theta_deg)
    
    # 1. Compute Bessel mode parameters (Zeros of J0)
    zn = jn_zeros(0, n_modes)
    omegan = (zn / 2.0) * np.sqrt(g / L)
    
    # 2. Calculate modal amplitudes An based on initial displacement
    An = (2 * L * theta0_rad) / (zn**2 * j1(zn))
    
    # 3. Superposition of modes for horizontal displacement (x) and velocity (vx)
    x_tip = np.zeros_like(t)
    vx_tip = np.zeros_like(t)
    
    for i in range(n_modes):
        cos_wt = np.cos(omegan[i] * t)
        sin_wt = np.sin(omegan[i] * t)
        x_tip += An[i] * cos_wt
        vx_tip -= An[i] * omegan[i] * sin_wt
        
    # 4. Geometric constraint correction to find vertical velocity (vy)
    # x^2 + y^2 = L^2  => vy = -(x * vx) / y
    y_tip = np.sqrt(np.maximum(L**2 - x_tip**2, 1e-9)) 
    vy_tip = - (x_tip * vx_tip) / y_tip
    
    # 5. Compute total velocity magnitude: v = sqrt(vx^2 + vy^2)
    v_total = np.sqrt(vx_tip**2 + vy_tip**2)
    
    # 6. Integrate velocity over time to find total path length (P)
    path_evolution = cumulative_trapezoid(v_total, t, initial=0)
    
    return path_evolution[-1]

# --- Execution of Multi-Angle Scan ---
L, g = 1.0, 9.8
duration = 200.0  # Time horizon matching FIG. 16
dt = 0.001        # Fine timestep to capture high-frequency Bessel modes
theta_list = [5, 15, 30, 90]

print(f"Calculating Theoretical Asymptotes for Continuum Limit...")
results = []
for theta in tqdm(theta_list):
    path = calculate_theoretical_asymptote(L, g, theta, duration, dt)
    results.append({'Theta_deg': theta, 'Theoretical_Path_m': path})

# --- Result Processing ---
df = pd.DataFrame(results)
# Linearity Check: If the system were perfectly linear, Path/Theta would be constant.
df['Linearity_Ratio'] = df['Theoretical_Path_m'] / df['Theta_deg']

print("\n--- Theoretical Continuum Limit Results ---")
print(df.to_string(index=False))

print("\n--- Linearity Maintenance Check (Path / Theta) ---")
print("Deviation in this ratio indicates the influence of nonlinear geometric correction.")
print(df[['Theta_deg', 'Linearity_Ratio']].to_string(index=False))