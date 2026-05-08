#!/usr/bin/env python3
"""
Manifold discovery with 1,000,000 tokens.
Takes 5-10 minutes to run.

Usage:
    uv run scripts/run_1m_tokens.py
"""

import sys
from pathlib import Path
import pickle

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
from transformer_lens import HookedTransformer
from manifold_analysis import extract_activations, find_manifolds, print_manifold_results


def main():
    # Detect CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA version: {torch.version.cuda}")
    
    print("Loading GPT-2 Small...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    
    # Extract activations
    X, tokens_list = extract_activations(
        model,
        target_token_count=1_000_000,
        target_layer=6
    )
    
    # Cluster
    from sklearn.cluster import MiniBatchKMeans
    print("Running MiniBatchKMeans with 500 clusters...")
    kmeans = MiniBatchKMeans(
        n_clusters=500,
        batch_size=2048,
        random_state=42,
        n_init="auto"
    )
    cluster_labels = kmeans.fit_predict(X)
    
    # Save extraction results for later analysis
    Path("Results").mkdir(exist_ok=True)
    np.savez_compressed(
        "Results/activations.npz",
        X=X,
        tokens_list=tokens_list,
        cluster_labels=cluster_labels
    )
    with open("Results/kmeans_model.pkl", "wb") as f:
        pickle.dump(kmeans, f)
    print("✓ Saved activations and cluster labels to Results/")
    
    # Find manifolds
    candidates = find_manifolds(
        X,
        tokens_list,
        num_clusters=500,
        variance_threshold=0.40,
        drop_off_threshold=1.5
    )
    
    # Print results
    print_manifold_results(candidates, top_n=5)


if __name__ == "__main__":
    main()
