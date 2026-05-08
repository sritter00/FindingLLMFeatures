#!/usr/bin/env python3
"""
Cardinal Directions PCA Analysis: PC1 vs PC3 Visualization
Shows the cool correlation between North-South and East-West directions.

Usage:
    uv run scripts/analyze_cardinal_pc1_pc3.py --layer 4
    uv run scripts/analyze_cardinal_pc1_pc3.py --layer 4 --tokens 2000
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


def visualize_pc1_vs_pc3(X: np.ndarray, tokens_list: list[str], 
                         layer: int, output_dir: str = "Results") -> None:
    """
    Create a detailed PC1 vs PC3 visualization showing N-S and E-W correlation.
    
    Args:
        X: Activation matrix [n_samples, d_model]
        tokens_list: Corresponding token strings
        layer: Layer number used for analysis
        output_dir: Directory to save output
    """
    print("\n=== CARDINAL DIRECTIONS: PC1 vs PC3 ANALYSIS ===")
    
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
    
    # Create figure with PC1 vs PC3 plot
    plt.figure(figsize=(10, 8))
    
    # Add legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=color, markersize=10, label=dir_name)
                      for dir_name, color in [('North', 'red'), ('South', 'blue'), 
                                            ('East', 'green'), ('West', 'orange')]]
    
    # PC1 vs PC3 plot with larger markers
    for i, token in enumerate(tokens_list):
        color = color_map.get(token, 'gray')
        plt.scatter(pca_results[i, 0], pca_results[i, 2], 
                   color=color, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    plt.legend(handles=legend_elements, loc='upper right', fontsize=12)
    plt.title(f"Cardinal Directions: PC1 vs PC3 (Layer {layer})", 
             fontweight='bold', fontsize=14)
    plt.xlabel(f"PC1 ({var_explained[0]:.1%} variance)", fontsize=12)
    plt.ylabel(f"PC3 ({var_explained[2]:.1%} variance)", fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add axis lines through origin for reference
    xlim = plt.xlim()
    ylim = plt.ylim()
    plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    plt.xlim(xlim)
    plt.ylim(ylim)
    
    plt.tight_layout()
    
    # Save figure
    Path(output_dir).mkdir(exist_ok=True)
    output_path = Path(output_dir) / f"cardinal_pc1_pc3_layer{layer}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to {output_path}")
    
    plt.show()
    
    # Print statistics
    print("\n=== CLUSTERING BY DIRECTION ===")
    unique_tokens, counts = np.unique(tokens_list, return_counts=True)
    for tok, count in zip(unique_tokens, counts):
        # Get mean position in PC1 vs PC3 space
        mask = np.array(tokens_list) == tok
        mean_pc1 = pca_results[mask, 0].mean()
        mean_pc3 = pca_results[mask, 2].mean()
        print(f"{tok:8s} (n={count:3d}): PC1={mean_pc1:+.3f}, PC3={mean_pc3:+.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze cardinal directions in PC1 vs PC3 space"
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
    
    # Visualize PC1 vs PC3
    visualize_pc1_vs_pc3(X, tokens_list, layer=args.layer)


if __name__ == "__main__":
    main()
