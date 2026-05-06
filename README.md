# Final Project Proposal: Discovering Multi-Dimensional Geometric Structures in LLM Feature Space

**Name:** Shane Ritter  
**Topic:** Interpretability / Mechanistic Geometry  
**Target Model:** GPT-2 Small  

---

## 1. Project Description
I plan to investigate whether Large Language Models (LLMs) represent complex conceptual relationships using specific geometric shapes beyond simple 1D linear vectors. While standard interpretability often focuses on "feature directions" (linear probes), recent work suggests that concepts like circularity (days of the week, seasons) or 2D grids (spatial coordinates) are encoded as multi-dimensional manifolds.

Instead of starting with a known concept (e.g., "North/South/East/West") and looking for its representation, I will implement a discovery-driven approach. I plan to use Principal Component Analysis (PCA) and Linear Probing across various transformer blocks of GPT-2 Small to identify clusters of activations that reside on non-linear planes. Specifically, I will scan the activation space for clusters that exhibit high variance in 2 or 3 dimensions but low variance in others, then use automated labeling (via a larger LLM or dictionary lookup) to determine what those features represent.

## 2. Literature Review
The foundation of this work is *"Not All Language Model Features Are One-Dimensionally Linear"* (Engels et al., 2024), which proved that models use circular representations for periodic concepts. My work also relates to *"The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets"* (Marks, Tegmark, et al., 2024), which discusses the Linear Representation Hypothesis.

Unlike the Engels paper, which manually selected concepts to test for circularity, I am inspired by the methodology of Sparse Autoencoders (Cunningham, et al., 2023), which seeks to find features without prior labeling. My project aims to bridge these two: using the unsupervised discovery of SAE-like analysis but focusing specifically on the geometric manifold (shapes) rather than just the sparse direction.

## 3. Hypothesis and Methodology

**Observations:** Many human concepts are inherently non-linear (e.g., the cyclical nature of time, the 2D nature of color wheels, or the hierarchical nature of taxonomies).

**Hypothesis:** GPT-2 Small encodes cyclical and relational data in distinct geometric planes (circles, rings, or lattices) that can be identified by analyzing the eigenvalues of activation clusters across transformer layers. I hypothesize that these "shapes" emerge more clearly in the middle-to-late layers of the model.

### Testing Plan
* **Extraction:** Use the `TransformerLens` or `nnsight` library in Python to extract residual stream activations from GPT-2 Small.
* **Manifold Search:** Apply a sliding window or clustering algorithm (like K-Means) to activations, followed by PCA on individual clusters. I will look for clusters where the first two principal components explain a similar, high percentage of variance (suggesting a 2D shape).
* **Visualization:** Project these clusters into 2D/3D space to visually confirm if they form circles, lines, or triangles.
* **Verification:** Use a "Linear Probe" vs. a "Circular Probe." If a circular probe (fitting to $\sin(\theta), \cos(\theta)$) has significantly lower loss than a linear probe for a specific feature cluster, I have successfully found a non-linear feature.

### Outcomes
* **Expected Outcome:** I expect to find circular representations for time-based concepts and potentially "star" or "hub-and-spoke" shapes for linguistic categories (e.g., a central verb connected to various tense conjugations).
* **Changing My Mind:** If every high-variance cluster I find is effectively captured by a single linear dimension (1D), I would conclude that the "Linear Representation Hypothesis" is more dominant than previously thought, and that multi-dimensional shapes are rare "edge cases" rather than a standard organizational principle.

## 4. Challenges and Risks
* **The "Needle in a Haystack" Problem:** The activation space is high-dimensional (768 for GPT-2 small). Finding a cluster that specifically forms a "shape" without knowing what the cluster represents beforehand is computationally difficult.
* **Labeling:** Once I find a "circle" in the math, identifying what that circle represents (e.g., "is this the months of the year or types of fruit?") will be difficult. I may need to use an LLM API to summarize the tokens that trigger those specific activations.
* **Likely Point of Failure:** I might find many "shapes" that turn out to be artifacts of the Softmax function or positional encodings rather than actual semantic concepts.
