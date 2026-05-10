import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import os

# Load your saved data files
# Update these paths to match YOUR actual file names
triple_data = np.load('triple_pendulum_results/optimized_data_20260504_013931.npz')  # ← Your triple pendulum file
double_data = np.load('double_pendulum_results_fast/double_pendulum_data_20260504_135305.npz')  # ← Your double pendulum file

# Extract data
triple_times = triple_data['chaos_times']
double_times = double_data['chaos_times']
triple_angles = triple_data['initial_angles']
double_angles = double_data['initial_angles']
triple_deviations = triple_data['angle_deviations']
double_deviations = double_data['angle_deviations']

t_span = 1800  # Max simulation time

# Calculate ratio
with np.errstate(divide='ignore', invalid='ignore'):
    ratio = np.where(double_times > 0, triple_times / double_times, np.nan)
    ratio = np.where(np.isinf(ratio), np.nan, ratio)

# Get actual min/max for scaling
valid_ratios = ratio[~np.isnan(ratio)]
actual_min = valid_ratios.min()
actual_max = valid_ratios.max()
actual_mean = valid_ratios.mean()
actual_median = np.median(valid_ratios)

print(f"Actual ratio range: {actual_min:.3f}x to {actual_max:.3f}x")
print(f"Mean: {actual_mean:.3f}x, Median: {actual_median:.3f}x")

# Create appropriate scale (add 10% padding)
scale_min = max(0.5, actual_min * 0.95)  # Don't go below 0.5
scale_max = actual_max * 1.05
mid_point = 1.0  # Equal performance

# Calculate symmetric range around 1.0
max_deviation = max(abs(scale_min - 1.0), abs(scale_max - 1.0))
symmetric_min = 1.0 - max_deviation
symmetric_max = 1.0 + max_deviation

print(f"\nUsing scale: {symmetric_min:.3f}x to {symmetric_max:.3f}x")

# Create comparison figure
fig = plt.figure(figsize=(20, 14))

# --- Row 1: Individual heatmaps ---
# Triple pendulum
ax1 = plt.subplot(2, 3, 1)
im1 = ax1.imshow(triple_times.T, 
                 extent=[0, 40, 0, 1.0],
                 origin='lower', aspect='auto', cmap='viridis',
                 vmin=0, vmax=t_span)
ax1.set_xlabel('Initial Angle (degrees)', fontsize=12)
ax1.set_ylabel('Deviation Angle (degrees)', fontsize=12)
ax1.set_title('Triple Pendulum\nChaos Time (seconds)', fontsize=12)
plt.colorbar(im1, ax=ax1)

# Double pendulum
ax2 = plt.subplot(2, 3, 2)
im2 = ax2.imshow(double_times.T, 
                 extent=[0, 40, 0, 1.0],
                 origin='lower', aspect='auto', cmap='viridis',
                 vmin=0, vmax=t_span)
ax2.set_xlabel('Initial Angle (degrees)', fontsize=12)
ax2.set_ylabel('Deviation Angle (degrees)', fontsize=12)
ax2.set_title('Double Pendulum\nChaos Time (seconds)', fontsize=12)
plt.colorbar(im2, ax=ax2)

# Difference (Triple - Double)
ax3 = plt.subplot(2, 3, 3)
difference = triple_times - double_times
max_diff = max(abs(difference.min()), abs(difference.max()))
im3 = ax3.imshow(difference.T, 
                 extent=[0, 40, 0, 1.0],
                 origin='lower', aspect='auto', cmap='RdBu_r',
                 vmin=-max_diff, vmax=max_diff)
ax3.set_xlabel('Initial Angle (degrees)', fontsize=12)
ax3.set_ylabel('Deviation Angle (degrees)', fontsize=12)
ax3.set_title(f'Difference (Triple - Double)\nRange: {difference.min():.0f}s to {difference.max():.0f}s', fontsize=12)
plt.colorbar(im3, ax=ax3, label='Time difference (seconds)')

# --- Row 2: Log scale comparison ---
# Triple log scale
ax4 = plt.subplot(2, 3, 4)
triple_log = triple_times.copy().astype(float)
triple_log[triple_log >= t_span] = np.nan
cmap_log = plt.cm.plasma.copy()
cmap_log.set_bad(color='white')
triple_chaotic = triple_times[triple_times < t_span]
triple_min = triple_chaotic.min() if len(triple_chaotic) > 0 else 0.1
im4 = ax4.imshow(triple_log.T, 
                 extent=[0, 40, 0, 1.0],
                 origin='lower', aspect='auto', cmap=cmap_log,
                 norm=colors.LogNorm(vmin=triple_min, vmax=t_span))
ax4.set_xlabel('Initial Angle (degrees)', fontsize=12)
ax4.set_ylabel('Deviation Angle (degrees)', fontsize=12)
ax4.set_title('Triple Pendulum\nLog Scale (White = Stable)', fontsize=12)
plt.colorbar(im4, ax=ax4)

# Double log scale
ax5 = plt.subplot(2, 3, 5)
double_log = double_times.copy().astype(float)
double_log[double_log >= t_span] = np.nan
double_chaotic = double_times[double_times < t_span]
double_min = double_chaotic.min() if len(double_chaotic) > 0 else 0.1
im5 = ax5.imshow(double_log.T, 
                 extent=[0, 40, 0, 1.0],
                 origin='lower', aspect='auto', cmap=cmap_log,
                 norm=colors.LogNorm(vmin=double_min, vmax=t_span))
ax5.set_xlabel('Initial Angle (degrees)', fontsize=12)
ax5.set_ylabel('Deviation Angle (degrees)', fontsize=12)
ax5.set_title('Double Pendulum\nLog Scale (White = Stable)', fontsize=12)
plt.colorbar(im5, ax=ax5)

# --- RATIO PLOT WITH APPROPRIATE SCALE (0.5 to 1.1) ---
ax6 = plt.subplot(2, 3, 6)

# Use linear scale with narrow range for maximum sensitivity
im6 = ax6.imshow(ratio.T, 
                 extent=[0, 40, 0, 1.0],
                 origin='lower', aspect='auto', 
                 cmap='coolwarm',
                 vmin=symmetric_min, vmax=symmetric_max,
                 interpolation='bilinear')

ax6.set_xlabel('Initial Angle (degrees)', fontsize=12)
ax6.set_ylabel('Deviation Angle (degrees)', fontsize=12)
ax6.set_title(f'Chaos Speed Ratio (Triple / Double)\nRange: {symmetric_min:.2f}x to {symmetric_max:.2f}x\nBlue=Double faster, Red=Triple faster', fontsize=10)

# Create detailed colorbar
cbar6 = plt.colorbar(im6, ax=ax6, label='Speed Ratio (Triple/Double)')
cbar6.set_ticks(np.linspace(symmetric_min, symmetric_max, 9))
cbar6.set_ticklabels([f'{x:.2f}x' for x in np.linspace(symmetric_min, symmetric_max, 9)])

# Add horizontal line at 1.0 (equal performance)
ax6.axhline(y=0.5, color='black', linewidth=0.5, alpha=0.3)
ax6.axvline(x=20, color='black', linewidth=0.5, alpha=0.3)

plt.suptitle('Double vs Triple Pendulum Chaos Comparison', fontsize=16, fontweight='bold')
plt.tight_layout()

# Save comparison figure
comparison_file = 'pendulum_comparison_optimized_scale.png'
plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
print(f"\n✅ Comparison saved as '{comparison_file}'")

# --- Create EXTRA PLOTS for better visualization ---
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Ratio with optimized scale
ax7 = axes2[0, 0]
im7 = ax7.imshow(ratio.T, 
                 extent=[0, 40, 0, 1.0],
                 origin='lower', aspect='auto', 
                 cmap='coolwarm',
                 vmin=symmetric_min, vmax=symmetric_max)
ax7.set_xlabel('Initial Angle (degrees)')
ax7.set_ylabel('Deviation Angle (degrees)')
ax7.set_title(f'Ratio (Triple/Double)\nScale: {symmetric_min:.2f}x to {symmetric_max:.2f}x', fontsize=12)
cbar7 = plt.colorbar(im7, ax=ax7, label='Speed Ratio')
cbar7.set_ticks([symmetric_min, 0.6, 0.7, 0.8, 0.9, 1.0, 1.05, symmetric_max])
cbar7.set_ticklabels([f'{symmetric_min:.2f}x', '0.60x', '0.70x', '0.80x', '0.90x', '1.00x', '1.05x', f'{symmetric_max:.2f}x'])

# Plot 2: Percentage difference (more intuitive)
ax8 = axes2[0, 1]
percent_diff = (ratio - 1.0) * 100
percent_min = (symmetric_min - 1.0) * 100
percent_max = (symmetric_max - 1.0) * 100
im8 = ax8.imshow(percent_diff.T, 
                 extent=[0, 40, 0, 1.0],
                 origin='lower', aspect='auto', 
                 cmap='coolwarm',
                 vmin=percent_min, vmax=percent_max)
ax8.set_xlabel('Initial Angle (degrees)')
ax8.set_ylabel('Deviation Angle (degrees)')
ax8.set_title(f'Percentage Difference\n(Triple - Double)/Double × 100%\nRange: {percent_min:.1f}% to {percent_max:.1f}%', fontsize=12)
cbar8 = plt.colorbar(im8, ax=ax8, label='Percentage (%)')
cbar8.set_ticks([percent_min, -20, -10, 0, 10, percent_max])
cbar8.set_ticklabels([f'{percent_min:.1f}%', '-20%', '-10%', '0%', '+10%', f'{percent_max:.1f}%'])

# Plot 3: Categorized regions (easy for project report)
ax9 = axes2[1, 0]
categories = np.zeros_like(ratio)
categories[ratio < 0.9] = 1    # Double faster (>10% difference)
categories[(ratio >= 0.9) & (ratio < 0.95)] = 2  # Double slightly faster
categories[(ratio >= 0.95) & (ratio <= 1.05)] = 3  # Similar (±5%)
categories[(ratio > 1.05) & (ratio <= 1.1)] = 4   # Triple slightly faster
categories[ratio > 1.1] = 5    # Triple faster (>10% difference)
categories[np.isnan(ratio)] = 0  # No data (stable regions)

# Define colormap for categories
category_cmap = plt.cm.colors.ListedColormap(['gray', '#2166ac', '#92c5de', '#f7f7f7', '#f4a582', '#b2182b'])
bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
norm = plt.cm.colors.BoundaryNorm(bounds, category_cmap.N)

im9 = ax9.imshow(categories.T, 
                 extent=[0, 40, 0, 1.0],
                 origin='lower', aspect='auto', 
                 cmap=category_cmap,
                 norm=norm)
ax9.set_xlabel('Initial Angle (degrees)')
ax9.set_ylabel('Deviation Angle (degrees)')
ax9.set_title('Chaos Speed Categories\n(For Project Interpretation)', fontsize=12)

# Add legend for categories
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='gray', label='Stable Region'),
    Patch(facecolor='#2166ac', label='Double FASTER (Triple <0.9x)'),
    Patch(facecolor='#92c5de', label='Double slightly faster (0.9-0.95x)'),
    Patch(facecolor='#f7f7f7', edgecolor='black', label='Similar speed (±5%)'),
    Patch(facecolor='#f4a582', label='Triple slightly faster (1.05-1.1x)'),
    Patch(facecolor='#b2182b', label='Triple FASTER (>1.1x)')
]
ax9.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=9)

# Plot 4: Difference heatmap (simpler interpretation)
ax10 = axes2[1, 1]
im10 = ax10.imshow(difference.T, 
                   extent=[0, 40, 0, 1.0],
                   origin='lower', aspect='auto', 
                   cmap='RdBu_r',
                   vmin=-max_diff, vmax=max_diff)
ax10.set_xlabel('Initial Angle (degrees)')
ax10.set_ylabel('Deviation Angle (degrees)')
ax10.set_title(f'Time Difference (Triple - Double)\nBlue=Double slower, Red=Triple slower\nRange: {difference.min():.0f}s to {difference.max():.0f}s', fontsize=12)
cbar10 = plt.colorbar(im10, ax=ax10, label='Time Difference (seconds)')

plt.suptitle(f'Triple vs Double Pendulum - Detailed Analysis\nRatio Range: {actual_min:.3f}x to {actual_max:.3f}x', fontsize=14, fontweight='bold')
plt.tight_layout()

extra_file = 'pendulum_comparison_optimized_enhanced.png'
plt.savefig(extra_file, dpi=300, bbox_inches='tight')
print(f"✅ Enhanced optimized comparison saved as '{extra_file}'")

plt.show()

# --- Statistics comparison ---
print("\n" + "="*70)
print("📊 DETAILED STATISTICAL COMPARISON")
print("="*70)

# Chaos rates
triple_chaotic_pct = np.sum(triple_times < t_span) / triple_times.size * 100
double_chaotic_pct = np.sum(double_times < t_span) / double_times.size * 100

print(f"Chaotic region percentage:")
print(f"  Triple pendulum: {triple_chaotic_pct:.1f}%")
print(f"  Double pendulum: {double_chaotic_pct:.1f}%")
print(f"  Difference: {triple_chaotic_pct - double_chaotic_pct:.1f}%")

# Average chaos times
triple_chaotic = triple_times[triple_times < t_span]
double_chaotic = double_times[double_times < t_span]

if len(triple_chaotic) > 0 and len(double_chaotic) > 0:
    print(f"\nAverage chaos time (chaotic region only):")
    print(f"  Triple pendulum: {triple_chaotic.mean():.1f} seconds")
    print(f"  Double pendulum: {double_chaotic.mean():.1f} seconds")
    print(f"  Triple is {triple_chaotic.mean()/double_chaotic.mean():.3f}x slower")

# Ratio statistics
print(f"\nSpeed Ratio (Triple/Double) statistics:")
print(f"  Mean ratio: {actual_mean:.3f}x")
print(f"  Median ratio: {actual_median:.3f}x")
print(f"  Min ratio: {actual_min:.3f}x")
print(f"  Max ratio: {actual_max:.3f}x")
print(f"  Std deviation: {valid_ratios.std():.3f}")

# Category breakdown (using the 5% threshold)
n_total = len(valid_ratios)
n_double_faster = np.sum(valid_ratios < 0.95)
n_similar = np.sum((valid_ratios >= 0.95) & (valid_ratios <= 1.05))
n_triple_faster = np.sum(valid_ratios > 1.05)

print(f"\nCategory breakdown (using 5% difference threshold):")
print(f"  Double faster (<0.95x): {n_double_faster} ({n_double_faster/n_total*100:.1f}%)")
print(f"  Similar speed (0.95x-1.05x): {n_similar} ({n_similar/n_total*100:.1f}%)")
print(f"  Triple faster (>1.05x): {n_triple_faster} ({n_triple_faster/n_total*100:.1f}%)")

print("\n" + "="*70)