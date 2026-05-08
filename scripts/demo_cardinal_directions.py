#!/usr/bin/env python3
"""
Demonstrate circular PCA on cardinal directions.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformer_lens import HookedTransformer
from circular_pca import visualize_cardinal_directions


def main():
    print("Loading GPT-2 Small...")
    model = HookedTransformer.from_pretrained("gpt2-small")

    print("Visualizing cardinal directions...")
    activations, pca_model, labels = visualize_cardinal_directions(
        model,
        target_layer=6
    )

    print("Cardinal directions visualization complete!")


if __name__ == "__main__":
    main()