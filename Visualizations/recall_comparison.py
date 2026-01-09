import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import glob

# Define the models and their paths
btl_models = {
    'ViTB-DN-GGeM': 'logs_beta/btl_cub200_1102_092709/RCIR/t608rp5s/checkpoints/test_csv',
    'ViTB-DN-GeM': 'logs_beta/btl_cub200_1103_112146/RCIR/1lsurln3/checkpoints/test_csv',
    'R50-GEM': 'logs_beta/btl_cub200_1103_022026/RCIR/a6amtgba/checkpoints/test_csv'
}

mcd_models = {
    'ViTB-DN-GGeM': 'logs_beta/mcd_cub200_1028_172537/RCIR/ldvrlkq3/checkpoints/test_csv',
    'ViTB-DN-GeM': 'logs_beta/mcd_cub200_1029_225432/RCIR/97iunl69/checkpoints/test_csv',
    'R50-GEM': 'logs_beta/mcd_cub200_1030_095639/RCIR/tlywtsk0/checkpoints/test_csv'
}

def get_recall_metrics(base_path):
    """Get all recall@1 values from test_csv subdirectories"""
    recalls = []
    
    # Find all metrics.csv files in subdirectories
    pattern = os.path.join(base_path, '*/metrics.csv')
    csv_files = glob.glob(pattern)
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if 'test_recall@1' in df.columns:
                recall_val = df['test_recall@1'].values[0]
                recalls.append(recall_val)
                print(f"  Found: {os.path.basename(os.path.dirname(csv_file))} -> {recall_val:.4f}")
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
    
    return recalls if recalls else []

# Collect data for BTL
btl_data = {}
print("\n=== BTL Models ===")
for model_name, path in btl_models.items():
    print(f"\n{model_name}:")
    recalls = get_recall_metrics(path)
    if recalls:
        btl_data[model_name] = recalls
        print(f"  -> {len(recalls)} trials, Mean: {np.mean(recalls):.4f}, Std: {np.std(recalls):.4f}")

# Collect data for MCD
mcd_data = {}
print("\n=== MCD Models ===")
for model_name, path in mcd_models.items():
    print(f"\n{model_name}:")
    recalls = get_recall_metrics(path)
    if recalls:
        mcd_data[model_name] = recalls
        print(f"  -> {len(recalls)} trials, Mean: {np.mean(recalls):.4f}, Std: {np.std(recalls):.4f}")

# Create the plot (side by side)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Colors for the bars (similar to the reference figure)
colors = ['#d62728', '#9467bd', '#17becf']  # Red, Purple, Cyan

# Plot BTL
ax1 = axes[0]
model_names = list(btl_data.keys())
means = [np.mean(btl_data[m]) for m in model_names]
stds = [np.std(btl_data[m]) for m in model_names]

y_pos = np.arange(len(model_names))
bars1 = ax1.barh(y_pos, means, xerr=stds, capsize=5, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
ax1.set_xlabel('Recall@1', fontsize=12)
ax1.set_title('BTL - IMFDB', fontsize=14, fontweight='bold')
ax1.set_yticks(y_pos)
ax1.set_yticklabels([])  # Remove y-axis labels
ax1.set_xlim([0.20, 0.95])
ax1.grid(axis='x', alpha=0.3, linestyle='--')
ax1.invert_yaxis()  # To have the first model at the top

# Add model names on bars and value labels
for i, (bar, mean, std, model_name, color) in enumerate(zip(bars1, means, stds, model_names, colors)):
    y_center = bar.get_y() + bar.get_height()/2. - 0.2
    
    # Add model name inside the bar, centered vertically and positioned aesthetically
    # Position at a safe location inside all bars
    x_name = 0.26
    ax1.text(x_name, y_center, model_name, ha='left', va='center', 
             fontsize=10, fontweight='bold', color='white',
             bbox=dict(boxstyle='round,pad=0.35', facecolor=color, edgecolor='none', alpha=0.95))
    
    # Add mean value label at the end
    ax1.text(mean + std + 0.02, y_center, f'{mean:.3f}', 
             ha='left', va='center', fontsize=9, fontweight='bold')
    
    # Add SD value on top of the error bar (at the end) - moved higher and more to the right
    ax1.text(mean + std + 0.02, y_center + 0.25, f'±{std:.3f}', 
             ha='left', va='bottom', fontsize=8, color='gray')

# Plot MCD
ax2 = axes[1]
model_names = list(mcd_data.keys())
means = [np.mean(mcd_data[m]) for m in model_names]
stds = [np.std(mcd_data[m]) for m in model_names]

y_pos = np.arange(len(model_names))
bars2 = ax2.barh(y_pos, means, xerr=stds, capsize=5, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
ax2.set_xlabel('Recall@1', fontsize=12)
ax2.set_title('MCD - IMFDB', fontsize=14, fontweight='bold')
ax2.set_yticks(y_pos)
ax2.set_yticklabels([])  # Remove y-axis labels
ax2.set_xlim([0.20, 0.95])
ax2.grid(axis='x', alpha=0.3, linestyle='--')
ax2.invert_yaxis()  # To have the first model at the top

# Add model names on bars and value labels
for i, (bar, mean, std, model_name, color) in enumerate(zip(bars2, means, stds, model_names, colors)):
    y_center = bar.get_y() + bar.get_height()/2. - 0.2
    
    # Add model name inside the bar, centered vertically and positioned aesthetically
    # Position at a safe location inside all bars
    x_name = 0.26
    ax2.text(x_name, y_center, model_name, ha='left', va='center', 
             fontsize=10, fontweight='bold', color='white',
             bbox=dict(boxstyle='round,pad=0.35', facecolor=color, edgecolor='none', alpha=0.95))
    
    # Add mean value label at the end
    ax2.text(mean + std + 0.02, y_center, f'{mean:.3f}', 
             ha='left', va='center', fontsize=9, fontweight='bold')
    
    # Add SD value on top of the error bar (at the end) - moved higher and more to the right
    ax2.text(mean + std + 0.02, y_center + 0.25, f'±{std:.3f}', 
             ha='left', va='bottom', fontsize=8, color='gray')

plt.tight_layout()
plt.savefig('comparison_BTL_MCD.png', dpi=300, bbox_inches='tight')
plt.savefig('comparison_BTL_MCD.pdf', bbox_inches='tight')
print("\nFigure saved as 'comparison_BTL_MCD.png' and 'comparison_BTL_MCD.pdf'")
plt.show()
