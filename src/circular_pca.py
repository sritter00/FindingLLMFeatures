"""Circular PCA visualization for studying cyclic concepts in neural networks."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer
from sklearn.decomposition import PCA


def visualize_circular_concept(
    model: HookedTransformer,
    concepts: list[str],
    prefix: str = "The current month is",
    target_layer: int = 6,
    save_path: str | None = None,
) -> tuple[np.ndarray, PCA, list[str]]:
    """
    Extract activations for cyclic concepts and visualize with PCA.
    
    Args:
        model: HookedTransformer model instance
        concepts: List of concept labels (e.g., months or days)
        prefix: Text prefix for the prompts
        target_layer: Which transformer layer to analyze
        save_path: Optional path to save the figure
    
    Returns:
        activation_matrix, pca_model, concept_labels
    """
    hook_name = f"blocks.{target_layer}.hook_resid_post"
    prompts = [f"{prefix} {concept}" for concept in concepts]
    
    # Extract activations
    print(f"Extracting residual stream activations at {hook_name}...")
    extracted_activations = []
    
    for prompt in prompts:
        logits, cache = model.run_with_cache(prompt)
        layer_activations = cache[hook_name]
        last_token_activation = layer_activations[0, -1, :].detach().cpu().numpy()
        extracted_activations.append(last_token_activation)
    
    activation_matrix = np.array(extracted_activations)
    
    # Apply PCA
    print("Applying PCA to find 2D geometric structure...")
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(activation_matrix)
    
    var_explained = pca.explained_variance_ratio_
    print(f"Variance explained by PC1: {var_explained[0]:.2%}")
    print(f"Variance explained by PC2: {var_explained[1]:.2%}")
    
    # Visualize
    plt.figure(figsize=(10, 8))
    plt.scatter(pca_results[:, 0], pca_results[:, 1], color='blue', s=100, alpha=0.7)
    
    for i, concept in enumerate(concepts):
        plt.annotate(concept, (pca_results[i, 0], pca_results[i, 1]), 
                     fontsize=12, xytext=(5, 5), textcoords='offset points')
    
    # Draw connecting lines for cyclic order
    for i in range(len(concepts)):
        start_idx = i
        end_idx = (i + 1) % len(concepts)
        plt.plot([pca_results[start_idx, 0], pca_results[end_idx, 0]], 
                 [pca_results[start_idx, 1], pca_results[end_idx, 1]], 
                 'k--', alpha=0.3)
    
    plt.title(f"PCA of GPT-2 Small Layer {target_layer} Residual Stream\nConcept: {concepts[0]} (Circular)")
    plt.xlabel(f"Principal Component 1 ({var_explained[0]:.1%} variance)")
    plt.ylabel(f"Principal Component 2 ({var_explained[1]:.1%} variance)")
    plt.grid(True, linestyle=':', alpha=0.6)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()
    
    return activation_matrix, pca, concepts
