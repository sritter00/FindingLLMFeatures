"""Manifold discovery in activation space using clustering and PCA."""

import torch
import numpy as np
from transformer_lens import HookedTransformer
from datasets import load_dataset
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from tqdm import tqdm
from typing import TypedDict


class ManifoldCandidate(TypedDict):
    """Represents a candidate 2D manifold in activation space."""
    cluster_id: int
    size: int
    var_2d: float
    drop_off: float
    sample_tokens: list[str]


def extract_activations(
    model: HookedTransformer,
    target_token_count: int = 100_000,
    target_layer: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Stream OpenWebText and extract layer activations.
    
    Args:
        model: HookedTransformer model instance
        target_token_count: Number of tokens to extract
        target_layer: Which transformer layer to analyze
    
    Returns:
        activation_matrix, token_strings
    """
    print("Loading dataset stream...")
    dataset = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    
    hook_name = f"blocks.{target_layer}.hook_resid_post"
    activation_buffer = []
    token_string_buffer = []
    
    print(f"Extracting activations for {target_token_count} tokens...")
    
    for row in tqdm(dataset):
        text = row['text']
        tokens = model.to_tokens(text, truncate=True)[:, :128]
        
        if tokens.shape[1] < 10:
            continue
        
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook_name)
        
        layer_acts = cache[hook_name][0].cpu().numpy()
        str_tokens = model.to_str_tokens(tokens[0])
        
        seen_tokens = set()
        for idx in range(1, len(str_tokens)):
            tok = str_tokens[idx].strip()  # Remove leading/trailing spaces
            if tok in seen_tokens:
                continue
            seen_tokens.add(tok)
            activation_buffer.append(layer_acts[idx])
            token_string_buffer.append(tok)
        
        if len(activation_buffer) >= target_token_count:
            break
    
    X = np.array(activation_buffer[:target_token_count])
    tokens_list = np.array(token_string_buffer[:target_token_count])
    
    print(f"Final Activation Matrix Shape: {X.shape} ({X.nbytes / 1e6:.2f} MB)")
    
    return X, tokens_list


def find_manifolds(
    X: np.ndarray,
    tokens_list: np.ndarray,
    num_clusters: int = 500,
    min_cluster_size: int = 50,
    variance_threshold: float = 0.40,
    drop_off_threshold: float = 1.5,
) -> list[ManifoldCandidate]:
    """
    Discover 2D manifolds in activation space using clustering and PCA.
    
    Args:
        X: Activation matrix [n_samples, d_model]
        tokens_list: Corresponding token strings
        num_clusters: Number of clusters to use
        min_cluster_size: Minimum cluster size to analyze
        variance_threshold: Minimum variance in PC1+PC2
        drop_off_threshold: Minimum ratio of PC2 to PC3
    
    Returns:
        List of manifold candidates, sorted by drop_off ratio
    """
    print(f"Running MiniBatchKMeans with {num_clusters} clusters...")
    kmeans = MiniBatchKMeans(
        n_clusters=num_clusters,
        batch_size=2048,
        random_state=42,
        n_init="auto"
    )
    cluster_labels = kmeans.fit_predict(X)
    
    print("Scanning clusters for 2D Manifolds...")
    manifold_candidates: list[ManifoldCandidate] = []
    
    for cluster_id in range(num_clusters):
        cluster_mask = (cluster_labels == cluster_id)
        cluster_acts = X[cluster_mask]
        cluster_tokens = tokens_list[cluster_mask]
        
        if len(cluster_acts) < min_cluster_size:
            continue
        
        pca = PCA(n_components=5)
        pca.fit(cluster_acts)
        var = pca.explained_variance_ratio_
        
        variance_in_2d = var[0] + var[1]
        drop_off = var[1] / (var[2] + 1e-9)
        
        if variance_in_2d > variance_threshold and drop_off > drop_off_threshold:
            # Get unique tokens to avoid duplicates in samples
            unique_tokens = np.unique(cluster_tokens)
            sample_size = min(10, len(unique_tokens))
            sample = np.random.choice(unique_tokens, sample_size, replace=False).tolist()
            
            manifold_candidates.append({
                'cluster_id': cluster_id,
                'size': len(cluster_acts),
                'var_2d': variance_in_2d,
                'drop_off': drop_off,
                'sample_tokens': sample
            })
    
    manifold_candidates.sort(key=lambda x: x['drop_off'], reverse=True)
    return manifold_candidates


def print_manifold_results(candidates: list[ManifoldCandidate], top_n: int = 5) -> None:
    """Print summary of top manifold candidates."""
    print("\n=== TOP 2D MANIFOLD CANDIDATES ===")
    
    for i, candidate in enumerate(candidates[:top_n]):
        print(f"\nCandidate {i+1} (Cluster {candidate['cluster_id']})")
        print(f"Size: {candidate['size']} tokens")
        print(f"Variance explained by PC1+PC2: {candidate['var_2d']:.2%}")
        print(f"Drop-off ratio (PC2/PC3): {candidate['drop_off']:.2f}x")
        print(f"Sample Tokens: {candidate['sample_tokens']}")
