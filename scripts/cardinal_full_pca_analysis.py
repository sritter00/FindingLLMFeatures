#!/usr/bin/env python3
"""
Cardinal Directions Full PCA Analysis
Shows all combinations of PC1, PC2, PC3 in 4 subplots (3 x 2D + 1 x 3D).

Usage:
    uv run scripts/cardinal_full_pca_analysis.py --layer 4
    uv run scripts/cardinal_full_pca_analysis.py --layer 4 --tokens 2000
"""

import sys
from pathlib import Path
import argparse

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
from transformer_lens import HookedTransformer
from datasets import load_dataset
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm


def find_cardinal_directions_batch(
    model: HookedTransformer,
    target_token_count: int = 50_000,
    target_layer: int = 4,
    batch_size: int = 8,
) -> tuple[np.ndarray, list[str]]:
    """
    Search for cardinal direction tokens in OpenWebText dataset using batch processing.
    """
    print("Loading dataset stream...")
    dataset = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    
    hook_name = f"blocks.{target_layer}.hook_resid_post"
    cardinal_tokens = {"north", "south", "east", "west"}
    
    activation_buffer = []
    token_buffer = []
    
    print(f"Searching for cardinal direction tokens ({target_token_count} total tokens)...")
    print(f"Using layer: {target_layer}, batch size: {batch_size}")
    
    batch_texts = []
    
    for row in tqdm(dataset):
        text = row['text']
        text_lower = text.lower()
        if not any(direction in text_lower for direction in cardinal_tokens):
            continue
        
        batch_texts.append(text)
        
        if len(batch_texts) >= batch_size:
            process_batch(batch_texts, model, hook_name, cardinal_tokens, 
                         activation_buffer, token_buffer, target_token_count)
            batch_texts = []
            
            if len(activation_buffer) >= target_token_count:
                break
    
    if batch_texts and len(activation_buffer) < target_token_count:
        process_batch(batch_texts, model, hook_name, cardinal_tokens,
                     activation_buffer, token_buffer, target_token_count)
    
    X = np.array(activation_buffer[:target_token_count])
    tokens_list = token_buffer[:target_token_count]
    
    print(f"Found {len(X)} cardinal direction tokens")
    if len(X) > 0:
        token_counts = dict(zip(*np.unique(tokens_list, return_counts=True)))
        print(f"Token distribution: {token_counts}")
    
    return X, tokens_list


def process_batch(texts, model, hook_name, cardinal_tokens, 
                 activation_buffer, token_buffer, target_token_count):
    """Process a batch of texts for cardinal directions."""
    for text in texts:
        if len(activation_buffer) >= target_token_count:
            return
            
        tokens = model.to_tokens(text, truncate=True)[:, :128]
        
        if tokens.shape[1] < 10:
            continue
        
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook_name)
        
        layer_acts = cache[hook_name][0].cpu().numpy()
        str_tokens = model.to_str_tokens(tokens[0])
        
        for idx in range(1, len(str_tokens)):
            if len(activation_buffer) >= target_token_count:
                return
                
            tok = str_tokens[idx].strip().lower()
            if tok in cardinal_tokens:
                activation_buffer.append(layer_acts[idx])
                token_buffer.append(str_tokens[idx])  # Keep original case


def visualize_full_pca(X: np.ndarray, tokens_list: list[str], 
                       layer: int, output_dir: str = "Results") -> None:
    """
    Create comprehensive 4-panel PCA visualization (PC1-PC2, PC2-PC3, PC1-PC3, 3D).
    
    Args:
        X: Activation matrix [n_samples, d_model]
        tokens_list: Corresponding token strings
        layer: Layer number used for analysis
        output_dir: Directory to save output
    """
    print("\n=== CARDINAL DIRECTIONS: FULL PCA ANALYSIS ===")
    
    # Apply PCA
    pca = PCA(n_components=3)
    pca_results = pca.fit_transform(X)
    
    var_explained = pca.explained_variance_ratio_
    print(f"Variance explained by PC1: {var_explained[0]:.2%}")
    print(f"Variance explained by PC2: {var_explained[1]:.2%}")
    print(f"Variance explained by PC3: {var_explained[2]:.2%}")
    
    # Color mapping for directions
    color_map = {
        'North': 'red',
        'north': 'red',
        'South': 'blue', 
        'south': 'blue',
        'East': 'green',
        'east': 'green',
        'West': 'orange',
        'west': 'orange'
    }
    
    # Create figure with 4 subplots (2x2)
    fig = plt.figure(figsize=(16, 14))
    
    # Legend elements (shared)
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=color, markersize=10, label=dir_name)
                      for dir_name, color in [('North', 'red'), ('South', 'blue'), 
                                            ('East', 'green'), ('West', 'orange')]]
    
    # Upper left: PC1 vs PC2
    ax1 = plt.subplot(2, 2, 1)
    for i, token in enumerate(tokens_list):
        color = color_map.get(token, 'gray')
        ax1.scatter(pca_results[i, 0], pca_results[i, 1], 
                   color=color, alpha=0.6, s=40, edgecolors='black', linewidth=0.5)
    
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=11)
    ax1.set_title(f"PC1 vs PC2", fontweight='bold', fontsize=12)
    ax1.set_xlabel(f"PC1 ({var_explained[0]:.1%})", fontsize=11)
    ax1.set_ylabel(f"PC2 ({var_explained[1]:.1%})", fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Upper right: PC2 vs PC3
    ax2 = plt.subplot(2, 2, 2)
    for i, token in enumerate(tokens_list):
        color = color_map.get(token, 'gray')
        ax2.scatter(pca_results[i, 1], pca_results[i, 2], 
                   color=color, alpha=0.6, s=40, edgecolors='black', linewidth=0.5)
    
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=11)
    ax2.set_title(f"PC2 vs PC3", fontweight='bold', fontsize=12)
    ax2.set_xlabel(f"PC2 ({var_explained[1]:.1%})", fontsize=11)
    ax2.set_ylabel(f"PC3 ({var_explained[2]:.1%})", fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Lower left: PC1 vs PC3
    ax3 = plt.subplot(2, 2, 3)
    for i, token in enumerate(tokens_list):
        color = color_map.get(token, 'gray')
        ax3.scatter(pca_results[i, 0], pca_results[i, 2], 
                   color=color, alpha=0.6, s=40, edgecolors='black', linewidth=0.5)
    
    ax3.legend(handles=legend_elements, loc='upper right', fontsize=11)
    ax3.set_title(f"PC1 vs PC3", fontweight='bold', fontsize=12)
    ax3.set_xlabel(f"PC1 ({var_explained[0]:.1%})", fontsize=11)
    ax3.set_ylabel(f"PC3 ({var_explained[2]:.1%})", fontsize=11)
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    # Lower right: 3D plot
    ax4 = plt.subplot(2, 2, 4, projection='3d')
    for i, token in enumerate(tokens_list):
        color = color_map.get(token, 'gray')
        ax4.scatter(pca_results[i, 0], pca_results[i, 1], pca_results[i, 2], 
                   color=color, alpha=0.6, s=40, edgecolors='black', linewidth=0.5)
    
    ax4.legend(handles=legend_elements, loc='upper right', fontsize=11)
    ax4.set_title(f"3D PCA Space", fontweight='bold', fontsize=12)
    ax4.set_xlabel(f"PC1 ({var_explained[0]:.1%})")
    ax4.set_ylabel(f"PC2 ({var_explained[1]:.1%})")
    ax4.set_zlabel(f"PC3 ({var_explained[2]:.1%})")
    
    plt.suptitle(f'Cardinal Directions: Full PCA Analysis (Layer {layer})', 
                fontweight='bold', fontsize=16, y=0.995)
    plt.tight_layout()
    
    # Save figure
    Path(output_dir).mkdir(exist_ok=True)
    output_path = Path(output_dir) / f"cardinal_full_pca_layer{layer}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to {output_path}")
    
    plt.show()
    
    # Print statistics
    print("\n=== CLUSTERING BY DIRECTION ===")
    unique_tokens, counts = np.unique(tokens_list, return_counts=True)
    print(f"\n{'Token':<10} {'Count':<8} {'PC1':<10} {'PC2':<10} {'PC3':<10}")
    print("─" * 50)
    for tok, count in zip(unique_tokens, counts):
        mask = np.array(tokens_list) == tok
        mean_pc1 = pca_results[mask, 0].mean()
        mean_pc2 = pca_results[mask, 1].mean()
        mean_pc3 = pca_results[mask, 2].mean()
        print(f"{tok:<10} {count:<8} {mean_pc1:<+10.3f} {mean_pc2:<+10.3f} {mean_pc3:<+10.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Full PCA analysis of cardinal directions"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=4,
        help="Transformer layer to analyze (default: 4)"
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=1_000,
        help="Number of cardinal direction tokens to collect (default: 1000)"
    )
    
    args = parser.parse_args()
    
    print("Checking for CUDA...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA version: {torch.version.cuda}")
    
    print("Loading GPT-2 Small...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    
    # Find cardinal direction tokens
    X, tokens_list = find_cardinal_directions_batch(
        model,
        target_token_count=args.tokens,
        target_layer=args.layer,
        batch_size=16
    )
    
    if len(X) == 0:
        print("No cardinal direction tokens found!")
        return
    
    # Visualize full PCA
    visualize_full_pca(X, tokens_list, layer=args.layer)


if __name__ == "__main__":
    main()
