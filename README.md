# Discovering Multi-Dimensional Geometric Structures in LLM Feature Space

**Name:** Shane Ritter
**Topic:** Interpretability / Mechanistic Geometry
**Target Model:** GPT-2 Small

---

## Final Project Report
**FULL REPORT:** [Final Project Report- Shane Ritter.pdf](./FinalReport/Final%20Project%20Report-%20Shane%20Ritter.pdf)

---
## SUMMARY OF REPORT
## 1. Project Overview
This project investigates the latent geometry of Large Language Models (LLMs) to identify low-dimensional manifolds within the activation space of **gpt2-small**. While traditional interpretability often focuses on 1D linear "feature directions," this research sought to discover if the model organizes related linguistic concepts into coherent, multi-dimensional geometric shapes.

Through a combination of unsupervised clustering and Principal Component Analysis (PCA), we isolated specific token groups—ranging from punctuation to semantic prepositions—that exhibit high-variance 2D structures.

## 2. Summary of Findings
The project successfully demonstrated that `gpt2-small` activation space is a structured environment where specific token classes occupy low-dimensional "sheets" rather than chaotic distributions.

### Key Discovery: Cardinal Directions (Layer 4)
The most significant finding emerged from a targeted investigation of **"North, South, East, and West"**. While automated clustering methods frequently overlook these features, manual PCA revealed a highly structured latent geometry:
* **Geometric Stacking**: We discovered that directional semantics are "stacked" in higher dimensions. While PC1 and PC2 capture general syntax, the directional correlation only becomes clear on **PC3**.
* **Opposites Trend**: Analysis showed that **North-South** tokens clustered together while **East-West** tokens formed a separate grouping, with both pairs demonstrating a clear trend of opposites across the principal components.
* **Relative Coordinate System**: In 3D space, these four directions occupy distinct "corners" of a geometric shape, suggesting the model has internalized a relative coordinate system for spatial orientation.

### Automated Manifold Results (Layer 6)
Broad scans of 100k and 1M tokens through Layer 6 identified several syntactic manifolds:
* **Punctuation/Symbol Manifolds**: Clusters representing parenthetical structures and ending symbols (e.g., `.`, `)`, `]`) were identified as "flat" 2D manifolds with high variance drop-offs.
* **Semantic Prepositions**: We identified manifolds for prepositions like `in`, `During`, `Under`, and `Within`, where proximity in activation space mirrored semantic similarity.

## 3. Methodology
* **Data**: Used a streaming version of the **OpenWebText** dataset.
* **Extraction**: Residual stream activations were extracted using the `HookedTransformer` library.
* **Clustering**: Applied **MiniBatchKMeans** with 500 clusters to group activation vectors.
* **PCA Scoring**: Ranked clusters based on the variance explained by the first two principal components (PC1+PC2) and the drop-off ratio between PC2 and PC3.

## 4. Limitations & Future Work
The research highlighted that high variance alone does not guarantee interpretability; many automated clusters appeared sporadic or "noisy" without a clear semantic trend. Additionally, current findings are limited to the linear rotations provided by PCA.

**Future directions include:**
* **Scaling**: Testing this methodology on larger models like **GPT-2 Medium**.
* **Complex Classes**: Extending analysis to other closed systems like numerical sequences or months of the year.