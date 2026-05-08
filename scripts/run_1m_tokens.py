#!/usr/bin/env python3
"""
Manifold discovery with 1,000,000 tokens.
Takes 5-10 minutes to run.

Usage:
    uv run scripts/run_1m_tokens.py
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformer_lens import HookedTransformer
from manifold_analysis import extract_activations, find_manifolds, print_manifold_results


def main():
    print("Loading GPT-2 Small...")
    model = HookedTransformer.from_pretrained("gpt2-small")
    
    # Extract activations
    X, tokens_list = extract_activations(
        model,
        target_token_count=1_000_000,
        target_layer=6
    )
    
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
