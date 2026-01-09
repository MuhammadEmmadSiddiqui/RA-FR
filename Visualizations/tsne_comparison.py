import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from os.path import join

# Model paths
model_paths = {
    'ViTB-DN-GGeM': 'logs_beta/btl_cub200_1102_092709/RCIR/t608rp5s/checkpoints/embeddings_test.pickle',
    'ViTB-DN-GeM': 'logs_beta/btl_cub200_1103_112146/RCIR/1lsurln3/checkpoints/embeddings_test.pickle',
    'R50-GeM': 'logs_beta/btl_cub200_1103_022026/RCIR/a6amtgba/checkpoints/embeddings_test.pickle'
}

# Load embeddings from all models
embeddings_dict = {}
for model_name, path in model_paths.items():
    with open(path, 'rb') as f:
        embeddings_dict[model_name] = pickle.load(f)
    print(f"{model_name}: Shape = {embeddings_dict[model_name].shape}")

# Load labels from CUB200 dataset
data_path = 'dbs/CUB_200_2011'
image_label = []
with open(join(data_path, "image_class_labels.txt"), 'r') as f:
    lines = f.readlines()
    for line in lines:
        label = int(line.split(' ')[1].strip())
        image_label.append(label)
image_label = np.array(image_label)

# Get test split labels (classes > 50 for CUB200)
test_indices = np.where(image_label > 50)[0]
test_labels = image_label[test_indices]

print(f"\nTest set size: {len(test_labels)}")
print(f"Unique classes in test: {len(np.unique(test_labels))}")
print(f"Class range: {test_labels.min()} to {test_labels.max()}")

# Select subset of classes for better visualization (e.g., 5 classes)
num_classes_to_plot = 5
unique_classes = np.unique(test_labels)
# Select evenly spaced classes
selected_classes = unique_classes[::len(unique_classes)//num_classes_to_plot][:num_classes_to_plot]
print(f"\nSelected classes for visualization: {selected_classes}")

# Filter data for selected classes
mask = np.isin(test_labels, selected_classes)
filtered_labels = test_labels[mask]

# Create color palette
colors = plt.cm.tab10(np.linspace(0, 1, num_classes_to_plot))
class_to_color = {cls: colors[i] for i, cls in enumerate(selected_classes)}

# Create figure with subplots - all in one row
fig, axes = plt.subplots(1, 3, figsize=(24, 7))

# Apply t-SNE to each model
tsne_results = {}
for idx, (model_name, embeddings) in enumerate(embeddings_dict.items()):
    print(f"\nApplying t-SNE to {model_name}...")
    
    # Filter embeddings for selected classes
    filtered_embeddings = embeddings[mask]
    
    # Standardize features
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(filtered_embeddings)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000, verbose=1)
    tsne_result = tsne.fit_transform(embeddings_scaled)
    tsne_results[model_name] = tsne_result
    
    # Plot
    ax = axes[idx]
    for cls in selected_classes:
        cls_mask = filtered_labels == cls
        ax.scatter(tsne_result[cls_mask, 0], 
                  tsne_result[cls_mask, 1],
                  c=[class_to_color[cls]], 
                  label=f'Class {cls}',
                  alpha=0.7,
                  s=80,
                  edgecolors='black',
                  linewidth=0.8)
    
    ax.set_title(f'{model_name}', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('t-SNE Component 1', fontsize=13)
    ax.set_ylabel('t-SNE Component 2', fontsize=13)
    ax.grid(True, alpha=0.3)

# Add single shared legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.0, 0.5), 
          fontsize=11, title='IMFDB', title_fontsize=12, frameon=True)

plt.tight_layout()
plt.savefig('tsne_3models_comparison.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Figure saved as 'tsne_3models_comparison.png'")
plt.show()

# Calculate and print some statistics
print("\n" + "="*60)
print("Summary Statistics:")
print("="*60)
for model_name, tsne_result in tsne_results.items():
    print(f"\n{model_name}:")
    print(f"  t-SNE shape: {tsne_result.shape}")
    print(f"  t-SNE range: X=[{tsne_result[:, 0].min():.2f}, {tsne_result[:, 0].max():.2f}], "
          f"Y=[{tsne_result[:, 1].min():.2f}, {tsne_result[:, 1].max():.2f}]")
