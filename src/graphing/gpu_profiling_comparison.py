import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Components (excluding aggregate totals)
components = [
    'Parameter Initialization',
    'Input Initialization',
    'Device MemAlloc',
    'Forward Propagation',
    'PCIe: Host to Device',
    'PCIe: Device to Host',
    'Output'
]

# Runtimes in seconds
runtimes = [
    0.00143423,
    8.2123,
    0.248333,
    24.6953,
    5.7236,
    4.52196,
    2.2
]

# Distinct colors: neutral blue for general ops, green/red for PCIe directions
colors = [
    '#1f77b4',  # Parameter Init
    '#1f77b4',  # Input Init
    '#1f77b4',  # Device MemAlloc
    '#1f77b4',  # Forward Prop
    '#2ca02c',  # PCIe: Host to Device (Green)
    '#d62728',  # PCIe: Device to Host (Red)
    '#1f77b4'   # Output
]

# Sort components by runtime duration for a cleaner layout
data = sorted(zip(components, runtimes, colors), key=lambda item: item[1])
sorted_components, sorted_runtimes, sorted_colors = zip(*data)

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Horizontal bar chart
bars = ax.barh(sorted_components, sorted_runtimes, color=sorted_colors, edgecolor='black', alpha=0.85)

# Annotate each bar with its exact runtime in seconds
max_runtime = max(sorted_runtimes)
for bar in bars:
    width = bar.get_width()
    # Display small values with more precision if needed, standard 4 decimals otherwise
    val_str = f'{width:.6f} s' if width < 0.01 else f'{width:.4f} s'
    ax.text(width + max_runtime * 0.015, bar.get_y() + bar.get_height() / 2, 
            val_str, ha='left', va='center', fontsize=10)

# Legend for colors
legend_elements = [
    Patch(facecolor='#1f77b4', edgecolor='black', label='General Operations'),
    Patch(facecolor='#2ca02c', edgecolor='black', label='PCIe: Host to Device'),
    Patch(facecolor='#d62728', edgecolor='black', label='PCIe: Device to Host')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

# Formatting
ax.set_xlabel('Runtime (seconds)', fontsize=12)
ax.set_ylabel('Component', fontsize=12)
ax.set_title('GPU Profiling Results Breakdown', fontsize=14, pad=12)
ax.grid(axis='x', linestyle='--', alpha=0.5)

# Add right-side padding for value text
ax.set_xlim(0, max_runtime * 1.18)

plt.tight_layout()

# Display interactive plot window
plt.show()