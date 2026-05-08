#!/usr/bin/env python3
"""
Search for cardinal direction tokens in OpenWebText and analyze their geometric structure.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
from transformer_lens import HookedTransformer
from datasets import load_dataset
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
from tqdm import tqdm


def find_cardinal_directions_batch(
    model: HookedTransformer,
    target_token_count: int = 50_000,
    target_layer: int = 6,
    batch_size: int = 8,
) -> tuple[np.ndarray, list[str]]:
    """
    Search for cardinal direction tokens in OpenWebText dataset using batch processing.
    
    Args:
        model: HookedTransformer model instance
        target_token_count: Number of tokens to extract
        target_layer: Which transformer layer to analyze
        batch_size: Number of texts to process simultaneously
    
    Returns:
        activation_matrix, token_strings for cardinal directions only
    """
    print("Loading dataset stream...")
    dataset = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    
    hook_name = f"blocks.{target_layer}.hook_resid_post"
    
    # Cardinal direction tokens we're looking for (case insensitive)
    cardinal_tokens = {"north", "south", "east", "west"}
    
    activation_buffer = []
    token_buffer = []
    
    print(f"Searching for cardinal direction tokens ({target_token_count} total tokens)...")
    print(f"Using batch size: {batch_size}")
    
    batch_texts = []
    
    for row in tqdm(dataset):
        text = row['text']
        
        # Quick pre-filter: check if text contains any cardinal directions
        text_lower = text.lower()
        if not any(direction in text_lower for direction in cardinal_tokens):
            continue
        
        batch_texts.append(text)
        
        # Process batch when it reaches batch_size
        if len(batch_texts) >= batch_size:
            process_batch(batch_texts, model, hook_name, cardinal_tokens, 
                         activation_buffer, token_buffer, target_token_count)
            batch_texts = []
            
            if len(activation_buffer) >= target_token_count:
                break
    
    # Process remaining texts
    if batch_texts and len(activation_buffer) < target_token_count:
        process_batch(batch_texts, model, hook_name, cardinal_tokens,
                     activation_buffer, token_buffer, target_token_count)
    
    X = np.array(activation_buffer[:target_token_count])
    tokens_list = token_buffer[:target_token_count]
    
    print(f"Found {len(X)} cardinal direction tokens")
    if len(X) > 0:
        print(f"Token distribution: {dict(zip(*np.unique(tokens_list, return_counts=True)))}")
    
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
        
        # Look for cardinal direction tokens
        for idx in range(1, len(str_tokens)):
            if len(activation_buffer) >= target_token_count:
                return
                
            tok = str_tokens[idx].strip().lower()
            if tok in cardinal_tokens:
                activation_buffer.append(layer_acts[idx])
                token_buffer.append(str_tokens[idx])  # Keep original case


def analyze_cardinal_geometry(X: np.ndarray, tokens_list: list[str]) -> None:
    """
    Analyze the geometric structure of cardinal direction activations.
    
    Args:
        X: Activation matrix [n_samples, d_model]
        tokens_list: Corresponding token strings
    """
    print("\n=== ANALYZING CARDINAL DIRECTION GEOMETRY ===")
    
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
    
    # Plot 2D PCA
    plt.figure(figsize=(12, 5))
    
    # 2D plot
    plt.subplot(1, 2, 1)
    for i, token in enumerate(tokens_list):
        color = color_map.get(token, 'gray')
        plt.scatter(pca_results[i, 0], pca_results[i, 1], color=color, alpha=0.6, s=30)
    
    # Add legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=color, markersize=8, label=dir_name)
                      for dir_name, color in [('North', 'red'), ('South', 'blue'), 
                                            ('East', 'green'), ('West', 'orange')]]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.title("Cardinal Directions in 2D PCA Space")
    plt.xlabel(f"PC1 ({var_explained[0]:.1%} variance)")
    plt.ylabel(f"PC2 ({var_explained[1]:.1%} variance)")
    plt.grid(True, alpha=0.3)
    
    # 3D plot
    ax = plt.subplot(1, 2, 2, projection='3d')
    for i, token in enumerate(tokens_list):
        color = color_map.get(token, 'gray')
        ax.scatter(pca_results[i, 0], pca_results[i, 1], pca_results[i, 2], 
                  color=color, alpha=0.6, s=30)
    
    ax.set_title("Cardinal Directions in 3D PCA Space")
    ax.set_xlabel(f"PC1 ({var_explained[0]:.1%})")
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1%})")
    ax.set_zlabel(f"PC3 ({var_explained[2]:.1%})")
    
    plt.tight_layout()
    plt.show()
    
    # Check for circular clustering
    print("\n=== CLUSTERING ANALYSIS ===")
    
    # K-means clustering to see if directions form distinct clusters
    kmeans = MiniBatchKMeans(n_clusters=4, random_state=42, n_init="auto")
    clusters = kmeans.fit_predict(X)
    
    # Analyze cluster composition
    cluster_composition = {}
    for cluster_id in range(4):
        cluster_mask = (clusters == cluster_id)
        cluster_tokens = [tokens_list[i] for i in range(len(tokens_list)) if cluster_mask[i]]
        token_counts = dict(zip(*np.unique(cluster_tokens, return_counts=True)))
        cluster_composition[cluster_id] = token_counts
    
    print("Cluster compositions:")
    for cluster_id, composition in cluster_composition.items():
        print(f"  Cluster {cluster_id}: {composition}")


def main():
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
        target_token_count=1_000,  # Collect 1k cardinal direction tokens
        target_layer=6,
        batch_size=16  # Process 16 texts at once
    )
    
    if len(X) == 0:
        print("No cardinal direction tokens found!")
        return
    
    # Analyze their geometric structure
    analyze_cardinal_geometry(X, tokens_list)


if __name__ == "__main__":
    main()