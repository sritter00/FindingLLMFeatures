#!/usr/bin/env python3
"""
Detailed PCA analysis of specific clusters found by run_1m_tokens.py

Usage:
    uv run scripts/analyze_cluster_pca.py --cluster 138 --data activations.npz
    uv run scripts/analyze_cluster_pca.py --cluster 138  # Re-extract if needed
"""

import sys
from pathlib import Path
import argparse
import pickle

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from transformer_lens import HookedTransformer
from manifold_analysis import extract_activations, find_manifolds
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA


def save_extraction_results(X, tokens_list, cluster_labels, kmeans, output_dir="Results"):
    """Save extraction and clustering results for later use."""
    Path(output_dir).mkdir(exist_ok=True)
    
    np.savez_compressed(
        f"{output_dir}/activations.npz",
        X=X,
        tokens_list=tokens_list,
        cluster_labels=cluster_labels
    )
    
    with open(f"{output_dir}/kmeans_model.pkl", "wb") as f:
        pickle.dump(kmeans, f)
    
    print(f"✓ Saved activations to {output_dir}/activations.npz")
    print(f"✓ Saved kmeans model to {output_dir}/kmeans_model.pkl")


def load_extraction_results(output_dir="Results", npz_file="activations.npz"):
    """Load previously saved extraction and clustering results."""
    npz_path = Path(output_dir) / npz_file
    data = np.load(npz_path)
    X = data['X']
    tokens_list = data['tokens_list']
    cluster_labels = data['cluster_labels']
    
    kmeans_file = npz_path.stem.replace('activations', 'kmeans_model')
    kmeans_path = Path(output_dir) / f"{kmeans_file}.pkl"
    with open(kmeans_path, "rb") as f:
        kmeans = pickle.load(f)
    
    print(f"✓ Loaded activations from {npz_path}")
    print(f"✓ Loaded kmeans model from {kmeans_path}")
    
    return X, tokens_list, cluster_labels, kmeans


def analyze_cluster_pca(
    X: np.ndarray,
    tokens_list: np.ndarray,
    cluster_labels: np.ndarray,
    cluster_id: int,
    num_components: int = 10,
    output_dir: str = "Results"
):
    """Perform detailed PCA analysis on a specific cluster."""
    
    # Extract cluster data
    cluster_mask = (cluster_labels == cluster_id)
    cluster_acts = X[cluster_mask]
    cluster_tokens = tokens_list[cluster_mask]
    
    print(f"\n{'='*60}")
    print(f"DETAILED PCA ANALYSIS: Cluster {cluster_id}")
    print(f"{'='*60}")
    print(f"Cluster size: {len(cluster_acts)} tokens")
    print(f"Activation dimensionality: {cluster_acts.shape[1]}")
    
    # Perform PCA
    pca = PCA(n_components=num_components)
    cluster_pca = pca.fit_transform(cluster_acts)
    
    # Print variance explained
    print(f"\n--- Variance Explained ---")
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    for i in range(min(5, len(pca.explained_variance_ratio_))):
        print(f"PC{i+1}: {pca.explained_variance_ratio_[i]:.2%} (cumsum: {cumsum[i]:.2%})")
    
    var_2d = pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1]
    drop_off = pca.explained_variance_ratio_[1] / (pca.explained_variance_ratio_[2] + 1e-9)
    print(f"\nVariance in PC1+PC2: {var_2d:.2%}")
    print(f"Drop-off ratio (PC2/PC3): {drop_off:.2f}x")
    
    # Get unique token frequencies
    unique_tokens, counts = np.unique(cluster_tokens, return_counts=True)
    sorted_indices = np.argsort(-counts)
    
    print(f"\nTop 20 tokens in cluster:")
    for i, idx in enumerate(sorted_indices[:20]):
        print(f"  {i+1}. '{unique_tokens[idx]}' ({counts[idx]} times)")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f'Cluster {cluster_id} - Detailed PCA Analysis', fontsize=16, fontweight='bold')
    
    # 1. Scatter plot: PC1 vs PC2
    ax = axes[0, 0]
    scatter = ax.scatter(cluster_pca[:, 0], cluster_pca[:, 1], 
                        c=range(len(cluster_acts)), cmap='viridis', alpha=0.6, s=20)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=11)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=11)
    ax.set_title('2D Manifold (PC1 vs PC2)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add 10 sample token labels
    sample_indices = np.random.choice(len(cluster_acts), min(10, len(cluster_acts)), replace=False)
    for idx in sample_indices:
        ax.annotate(cluster_tokens[idx], 
                   xy=(cluster_pca[idx, 0], cluster_pca[idx, 1]),
                   fontsize=8, alpha=0.7)
    
    # 2. Variance explained
    ax = axes[0, 1]
    ax.bar(range(1, 6), pca.explained_variance_ratio_[:5], alpha=0.7, color='steelblue')
    ax.set_xlabel('Principal Component', fontsize=11)
    ax.set_ylabel('Variance Explained', fontsize=11)
    ax.set_title('Variance Explained by PC', fontweight='bold')
    ax.set_xticks(range(1, 6))
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Cumulative variance
    ax = axes[1, 0]
    cumsum = np.cumsum(pca.explained_variance_ratio_[:10])
    ax.plot(range(1, 11), cumsum, 'o-', linewidth=2, markersize=6, color='darkgreen')
    ax.axhline(y=0.9, color='r', linestyle='--', label='90% threshold', alpha=0.7)
    ax.set_xlabel('Number of Components', fontsize=11)
    ax.set_ylabel('Cumulative Variance Explained', fontsize=11)
    ax.set_title('Cumulative Variance', fontweight='bold')
    ax.set_xticks(range(1, 11))
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 4. Top tokens on manifold
    ax = axes[1, 1]
    ax.axis('off')
    
    # Create text summary
    text_lines = [
        f"Cluster {cluster_id} Summary",
        "─" * 40,
        f"Size: {len(cluster_acts)} tokens",
        f"Dimensionality: {cluster_acts.shape[1]}",
        f"PC1+PC2 Variance: {var_2d:.2%}",
        f"Drop-off Ratio: {drop_off:.2f}x",
        "",
        "Top 10 Tokens:",
    ]
    
    for i, idx in enumerate(sorted_indices[:10]):
        text_lines.append(f"  {i+1}. '{unique_tokens[idx]}' ({counts[idx]}x)")
    
    text_summary = "\n".join(text_lines)
    ax.text(0.1, 0.95, text_summary, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_dir) / f"cluster_{cluster_id}_pca_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to {output_path}")
    
    plt.show()
    
    return pca, cluster_pca, unique_tokens, counts


def main():
    parser = argparse.ArgumentParser(
        description="Detailed PCA analysis of specific clusters"
    )
    parser.add_argument(
        "--cluster",
        type=int,
        required=True,
        help="Cluster ID to analyze"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="Results",
        help="Directory with saved activation data (default: Results/)"
    )
    parser.add_argument(
        "--npz-file",
        type=str,
        default="activations.npz",
        help="NPZ file name to load (default: activations.npz, use activations_100k.npz for 100k run)"
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Re-run extraction instead of loading saved data"
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=1_000_000,
        help="Number of tokens to extract (if --rerun)"
    )
    
    args = parser.parse_args()
    
    # Load or extract data
    data_dir = args.data
    npz_file = args.npz_file
    if args.rerun or not Path(f"{data_dir}/{npz_file}").exists():
        print("Extracting activations from GPT-2 Small...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        model = HookedTransformer.from_pretrained("gpt2-small", device=device)
        
        X, tokens_list = extract_activations(
            model,
            target_token_count=args.tokens,
            target_layer=6
        )
        
        # Cluster
        print(f"\nRunning MiniBatchKMeans with 500 clusters...")
        kmeans = MiniBatchKMeans(
            n_clusters=500,
            batch_size=2048,
            random_state=42,
            n_init="auto"
        )
        cluster_labels = kmeans.fit_predict(X)
        
        # Save for future use
        save_extraction_results(X, tokens_list, cluster_labels, kmeans, data_dir)
    else:
        X, tokens_list, cluster_labels, kmeans = load_extraction_results(data_dir, npz_file)
    
    # Analyze specific cluster
    analyze_cluster_pca(
        X, tokens_list, cluster_labels,
        cluster_id=args.cluster,
        num_components=10,
        output_dir=args.data
    )


if __name__ == "__main__":
    main()
