import numpy as np
import matplotlib.pyplot as plt
import os
import time

# Load your existing data
# Update this path to your actual data file
data_file = "explicit_pendulum_results/explicit_N7_data_20260505_145047.npz"  # ← Update with your filename

data = np.load(data_file, allow_pickle=True)
pendulum_counts = data['pendulum_counts']
deviations = data['deviations']
chaos_times = data['chaos_times']

# Create zoomed-in figure
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Color map for different pendulum counts
colors = plt.cm.viridis(np.linspace(0, 1, len(pendulum_counts)))

# Plot 1: Full range (for reference)
ax1 = axes[0]
for i, N in enumerate(pendulum_counts):
    ax1.plot(deviations, chaos_times[:, i], linewidth=2, 
             label=f'{N} pendulums', color=colors[i], alpha=0.8)
ax1.set_xlabel('Deviation Angle (degrees)', fontsize=14)
ax1.set_ylabel('Time to Chaos (seconds)', fontsize=14)
ax1.set_title('Full Range (0-1.0°, 0-1800s)', fontsize=12)
ax1.legend(fontsize=10, loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 1.0)
ax1.set_ylim(0, 1800)

# Plot 2: Zoomed in on fast chaos region
ax2 = axes[1]
for i, N in enumerate(pendulum_counts):
    ax2.plot(deviations, chaos_times[:, i], linewidth=2.5, 
             label=f'{N} pendulums', color=colors[i], alpha=0.9)
ax2.set_xlabel('Deviation Angle (degrees)', fontsize=14)
ax2.set_ylabel('Time to Chaos (seconds)', fontsize=14)
ax2.set_title('Zoomed: 0-0.3° Deviation, 0-600s (Fast Chaos Region)', fontsize=12)
ax2.legend(fontsize=10, loc='upper right')
ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax2.set_xlim(0, 0.3)
ax2.set_ylim(0, 600)


plt.suptitle(f'Multi-Pendulum Chaos Analysis - Fast Chaos Region\nInitial Angle: {data["initial_angle"][()]}°, Max Time: {data["t_span"][()]}s', 
             fontsize=14, fontweight='bold')
plt.tight_layout()

# Save the zoomed figure
timestamp = time.strftime("%Y%m%d_%H%M%S")
filename = os.path.join(os.path.dirname(data_file), f'zoomed_fast_chaos_{timestamp}.png')
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"✅ Zoomed figure saved as '{filename}'")

plt.show()

# Also create an inset plot for even more detail
fig2, ax = plt.subplots(figsize=(12, 8))

# Main zoomed plot
for i, N in enumerate(pendulum_counts):
    ax.plot(deviations, chaos_times[:, i], linewidth=2.5, 
             label=f'{N} pendulums', color=colors[i], alpha=0.9)
ax.set_xlabel('Deviation Angle (degrees)', fontsize=14)
ax.set_ylabel('Time to Chaos (seconds)', fontsize=14)
ax.set_title('Zoomed: 0-0.3° Deviation, 0-600s\n(Fast Chaos Onset Region)', fontsize=14)
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax.set_xlim(0, 0.3)
ax.set_ylim(0, 600)

# Save the detailed zoomed figure
filename2 = os.path.join(os.path.dirname(data_file), f'zoomed_detailed_{timestamp}.png')
plt.savefig(filename2, dpi=300, bbox_inches='tight')
print(f"✅ Detailed zoomed figure saved as '{filename2}'")

plt.show()


# Select indices for deviation < 0.3°
mask = deviations <= 0.3
dev_small = deviations[mask]

for i, N in enumerate(pendulum_counts):
    times_small = chaos_times[mask, i]
    
    # Count how many simulations are below 600s in this region
    below_600 = np.sum(times_small < 600)
    below_400 = np.sum(times_small < 400)
    below_200 = np.sum(times_small < 200)
    below_50 = np.sum(times_small < 50)
    
    total = len(times_small)
    
    print(f"\n{N}-Pendulum System (0-0.3° deviation):")
    print(f"  • Average chaos time: {np.mean(times_small):.1f}s")
    print(f"  • Min chaos time: {times_small.min():.1f}s")
    print(f"  • Chaos within 50s: {below_50}/{total} ({below_50/total*100:.1f}%)")
    print(f"  • Chaos within 200s: {below_200}/{total} ({below_200/total*100:.1f}%)")
    print(f"  • Chaos within 400s: {below_400}/{total} ({below_400/total*100:.1f}%)")
    print(f"  • Chaos within 600s: {below_600}/{total} ({below_600/total*100:.1f}%)")

print("\n" + "="*70)
print("Analysis complete! Check the saved PNG files.")
print("="*70)