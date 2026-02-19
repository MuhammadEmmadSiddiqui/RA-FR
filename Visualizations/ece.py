import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from os.path import join

# Configuration
dataset = "cub200"
n_bins = 10
k_at = 1  # Recall@1 for calibration
n_seeds = 10

# Define test configurations
test_configs = [
    {'name': 'IMFDB', 'method': 'BTL', 'base': 'figures/rcir/ECE/IMFDB/BTL'},
    {'name': 'IMFDB', 'method': 'MCD', 'base': 'figures/rcir/ECE/IMFDB/MCD'},
    {'name': 'SCFace', 'method': 'BTL', 'base': 'figures/rcir/ECE/SCFace/BTL'},
    {'name': 'SCFace', 'method': 'MCD', 'base': 'figures/rcir/ECE/SCFace/MCD'},
]

# Model names (same for all test configs)
model_names = ['ViT-D2-GGeM', 'ViT-D2-GeM', 'R50-GeM']

# Colors for each model
colors = {
    'ViT-D2-GGeM': 'coral',
    'ViT-D2-GeM': 'steelblue',
    'R50-GeM': 'gray'
}

def process_model_data(base_dir, model_name, dataset, n_seeds, k_at):
    """Process data for a single model"""
    output_dir = join(base_dir, model_name)
    all_recalls = []
    all_uncertainties = []

    for seed in range(n_seeds):
        file_path = join(output_dir, f'ece_{dataset}_{seed}.pickle')
        
        # Load all three objects from the pickle file
        with open(file_path, 'rb') as f:
            preds_test = pickle.load(f)      # (N, k) - predictions
            q_sigma_test = pickle.load(f)     # (N, 1) or (N,) - uncertainties
            positives_test = pickle.load(f)   # list of arrays - ground truth
        
        # Calculate per-query recall@k
        recalls = []
        uncertainties = []
        
        for i in range(len(preds_test)):
            # Check if any of top-k predictions are correct
            is_correct = np.in1d(preds_test[i, :k_at], positives_test[i]).sum() > 0
            recalls.append(float(is_correct))
            
            # Extract uncertainty
            unc = q_sigma_test[i]
            if isinstance(unc, np.ndarray):
                unc = unc[0] if len(unc.shape) > 0 else float(unc)
            uncertainties.append(unc)
        
        all_recalls.append(recalls)
        all_uncertainties.append(uncertainties)

    # Convert to numpy arrays
    all_recalls = np.array(all_recalls)  # (n_seeds, n_queries)
    all_uncertainties = np.array(all_uncertainties)  # (n_seeds, n_queries)

    # Normalize uncertainties to [0, 1] across all seeds
    unc_min = all_uncertainties.min()
    unc_max = all_uncertainties.max()
    all_uncertainties_norm = (all_uncertainties - unc_min) / (unc_max - unc_min)

    # Create bins based on uncertainty
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Calculate reliability curve (average across seeds)
    bin_recalls_mean = []
    bin_recalls_std = []
    bin_counts = []
    bin_uncertainties = []

    for i in range(n_bins):
        bin_recalls_per_seed = []
        bin_unc_per_seed = []
        
        for seed in range(n_seeds):
            # Find queries in this uncertainty bin for this seed
            mask = (all_uncertainties_norm[seed] >= bins[i]) & (all_uncertainties_norm[seed] < bins[i+1])
            
            if mask.sum() > 0:
                bin_recalls_per_seed.append(all_recalls[seed, mask].mean())
                bin_unc_per_seed.append(all_uncertainties_norm[seed, mask].mean())
        
        if len(bin_recalls_per_seed) > 0:
            bin_recalls_mean.append(np.mean(bin_recalls_per_seed))
            bin_recalls_std.append(np.std(bin_recalls_per_seed))
            bin_uncertainties.append(np.mean(bin_unc_per_seed))
            
            # Count from first seed
            mask = (all_uncertainties_norm[0] >= bins[i]) & (all_uncertainties_norm[0] < bins[i+1])
            bin_counts.append(mask.sum())
        else:
            bin_recalls_mean.append(0)
            bin_recalls_std.append(0)
            bin_uncertainties.append(bin_centers[i])
            bin_counts.append(0)

    bin_recalls_mean = np.array(bin_recalls_mean)
    bin_recalls_std = np.array(bin_recalls_std)
    bin_uncertainties = np.array(bin_uncertainties)

    # Calculate ECE
    ece = np.sum(np.array(bin_counts) * np.abs(bin_recalls_mean - (1 - bin_uncertainties))) / sum(bin_counts)
    
    return {
        'bin_recalls_mean': bin_recalls_mean,
        'bin_recalls_std': bin_recalls_std,
        'bin_uncertainties': bin_uncertainties,
        'bin_counts': bin_counts,
        'all_uncertainties_norm': all_uncertainties_norm,
        'ece': ece
    }

# Process all configurations
all_results = {}

for config in test_configs:
    config_key = f"{config['name']}_{config['method']}"
    print(f'\n{"="*60}')
    print(f'Processing {config_key}')
    print(f'{"="*60}')
    
    all_results[config_key] = {}
    
    for model_name in model_names:
        print(f'  Model: {model_name}')
        try:
            model_data = process_model_data(config['base'], model_name, dataset, n_seeds, k_at)
            all_results[config_key][model_name] = model_data
            print(f'    ECE: {model_data["ece"]:.4f}')
        except Exception as e:
            print(f'    Error: {e}')

# Create separate figures for IMFDB and SCFace
output_dir = 'figures/rcir/ECE'
os.makedirs(output_dir, exist_ok=True)

for dataset_name in ['IMFDB', 'SCFace']:
    # Create figure with 2x2 grid: calibration plots on top, density plots on bottom
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), gridspec_kw={'height_ratios': [1, 0.6], 'hspace': 0.25})
    
    # Top row: BTL and MCD calibration plots
    methods_order = ['BTL', 'MCD']
    
    for idx, method in enumerate(methods_order):
        ax = axes[0, idx]
        config_key = f"{dataset_name}_{method}"
        
        # Plot perfect calibration line
        ax.plot([0, 1], [1, 0], 'k--', alpha=0.5, linewidth=2, label='Perfect calibration')
        
        # Plot each model
        if config_key in all_results:
            for model_name in model_names:
                if model_name in all_results[config_key]:
                    data = all_results[config_key][model_name]
                    color = colors[model_name]
                    
                    ax.plot(data['bin_uncertainties'], data['bin_recalls_mean'], 
                           'o-', linewidth=2.5, markersize=8, color=color, 
                           label=f'{model_name} (ECE={data["ece"]:.3f})')
                    ax.fill_between(data['bin_uncertainties'], 
                                  np.clip(data['bin_recalls_mean'] - data['bin_recalls_std'], 0, 1),
                                  np.clip(data['bin_recalls_mean'] + data['bin_recalls_std'], 0, 1),
                                  alpha=0.2, color=color)
        
        # Formatting
        ax.set_ylabel('Recall@1', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.set_title(method, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
    
    # Bottom row: Separate density plots for BTL and MCD
    for idx, method in enumerate(methods_order):
        ax_density = axes[1, idx]
        config_key = f"{dataset_name}_{method}"
        
        if config_key in all_results:
            for model_name in model_names:
                if model_name in all_results[config_key]:
                    data = all_results[config_key][model_name]
                    color = colors[model_name]
                    
                    ax_density.hist(data['all_uncertainties_norm'][0], bins=30, alpha=0.5, 
                                   color=color, edgecolor='black', linewidth=0.5, 
                                   density=True, label=model_name)
        
        ax_density.set_xlabel('Uncertainty', fontsize=12, fontweight='bold')
        ax_density.set_ylabel('Density', fontsize=12, fontweight='bold')
        ax_density.set_xlim(0, 1)
        ax_density.grid(True, alpha=0.3, axis='y')
        ax_density.legend(fontsize=10, loc='best')
    
    # Add main title
    fig.suptitle(dataset_name, fontsize=18, fontweight='bold', y=0.98)
    
    # Save figure
    output_path = join(output_dir, f'calibration_{dataset}_{dataset_name}_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(join(output_dir, f'calibration_{dataset}_{dataset_name}_comparison.pdf'), bbox_inches='tight')
    print(f'\n{"="*60}')
    print(f'Plot saved to: {output_path}')
    print(f'{"="*60}')
    plt.show()
    plt.close()

# Print summary statistics
print('\n' + '='*60)
print('SUMMARY STATISTICS')
print('='*60)
for config in test_configs:
    config_key = f"{config['name']}_{config['method']}"
    print(f'\n{config_key}:')
    if config_key in all_results:
        for model_name in model_names:
            if model_name in all_results[config_key]:
                data = all_results[config_key][model_name]
                print(f'  {model_name}:')
                print(f'    ECE: {data["ece"]:.4f}')
                print(f'    Avg Recall@1: {data["bin_recalls_mean"].mean():.3f}')