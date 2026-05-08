# Finding LLM Features

Research project investigating latent features and manifolds in transformer language model representations.

## Project Structure

```
.
├── pyproject.toml              # uv project configuration
├── src/
│   ├── __init__.py
│   ├── circular_pca.py         # Circular concept visualization
│   └── manifold_analysis.py    # Manifold discovery utilities
├── scripts/
│   ├── run_100k_tokens.py      # Quick test (1-2 min)
│   ├── run_1m_tokens.py        # Full analysis (5-10 min)
│   ├── analyze_cluster_pca.py  # Detailed PCA analysis of specific clusters
│   └── analyze_cardinal_pc1_pc3.py  # Cardinal directions PC1 vs PC3 visualization
├── Results/                    # Generated activations & analysis outputs (gitignored)
├── StarterCode.ipynb           # Interactive notebook
└── README.md
```

## Quick Start with `uv`

### Installation

First, [install `uv`](https://docs.astral.sh/uv/):

```bash
# On Windows
curl -LsSf https://astral.sh/uv/install.ps1 | powershell -c -

# Or using pip
pip install uv
```

### Run Scripts

```bash
# Quick test (100k tokens, ~2-5 minutes with cpu, with cuda ~1 min)
uv run scripts/run_100k_tokens.py


### Analyze Clusters

After running the scripts above, analyze specific 2D manifold candidates:

```bash
# Analyze cluster 138 from 1M token run (prepositions)
uv run scripts/analyze_cluster_pca.py --cluster 138 --npz-file activations.npz

# Analyze cluster 297 from 100k token run (punctuation)
uv run scripts/analyze_cluster_pca.py --cluster 297 --npz-file activations_100k.npz

# Detailed cardinal directions analysis (PC1 vs PC3)
uv run scripts/analyze_cardinal_pc1_pc3.py --layer 4 --tokens 1000
```
# Full analysis (1M tokens, ~15-30 minutes with cpu, with cuda ~3-10 min [mine took ~5 min with NIVIDA 4080 Super])
uv run scripts/run_1m_tokens.py
```

### Use in Jupyter Notebook

```python
# In a notebook cell
from src.circular_pca import visualize_circular_concept
from src.manifold_analysis import extract_activations, find_manifolds


### `analyze_cluster_pca.py`
Performs detailed PCA analysis on specific clusters:
- Creates 4-panel visualization (PC1 vs PC2, variance plots, cumulative variance)
- Displays top tokens and their frequencies
- Calculates manifold metrics (variance, drop-off ratio)
- Supports both 100k and 1M token datasets
- `analyze_cluster_pca()` - Detailed PCA analysis of a specific cluster
- `visualize_pc1_vs_pc3()` - Cardinal directions visualization

## Large Files & Git

Activation data files (`.npz`, `.pkl`) are **gitignored** because they exceed 100MB. This is intentional:

- **Run scripts locally** to generate activation files as needed
- **Analysis outputs** (`.png` images) are small and can be committed
- **Results/** directory is excluded from git tracking

If you need to share large files, use cloud storage or regenerate locally.

### `analyze_cardinal_pc1_pc3.py`
Specialized visualization for cardinal direction tokens (North, South, East, West):
- Focuses on PC1 vs PC3 to show directional correlations
- Generates scatter plots with directional color coding
- Supports custom layer selection for exploring different depths
months = ["January", "February", "March", ..., "December"]
activations, pca, labels = visualize_circular_concept(model, months)
```

## What Each Module Does

### `circular_pca.py`
Extracts and visualizes cyclic concept representations (e.g., months, days) using PCA to show their circular structure in activation space.

### `manifold_analysis.py`
Searches for 2D manifolds in activation space by:
1. **Streaming tokens** from OpenWebText
2. **Extracting activations** from a specific layer
3. **Clustering** activations with K-Means
4. **Scoring clusters** for 2D structure (using PCA variance drop-off)

## Key Functions

- `visualize_circular_concept()` - PCA visualization of cyclic concepts
- `extract_activations()` - Stream and extract from OpenWebText
- `find_manifolds()` - Discover 2D structures in clusters
- `print_manifold_results()` - Display top candidates

## Dependencies

Automatically installed via `uv run`:
- `torch` - Deep learning
- `transformer-lens` - Model interpretation
- `datasets` - OpenWebText streaming
- `scikit-learn` - PCA, K-Means
- `matplotlib` - Visualization
- `numpy`, `tqdm` - Utilities
