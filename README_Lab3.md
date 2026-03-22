# MSCS 634 – Lab 3: Clustering Analysis Using K-Means and K-Medoids

**Course:** MSCS 634-B01 – Advanced Big Data and Data Mining (Spring, Second Bi-term)  
**Lab:** Lab 3 – Clustering Analysis Using K-Means and K-Medoids Algorithms

---

## Purpose

This lab applies two unsupervised clustering algorithms — **K-Means** and **K-Medoids** — to the Wine Dataset from scikit-learn. The dataset contains 178 wine samples across three cultivar classes, described by 13 chemical properties (alcohol, malic acid, flavanoids, etc.).

The goals are to:
1. Implement and compare K-Means and K-Medoids clustering with k = 3.
2. Evaluate cluster quality using the **Silhouette Score** and **Adjusted Rand Index (ARI)**.
3. Visualize clusters using PCA projection and compare against true class labels.
4. Determine when each algorithm is preferable based on data characteristics.

---

## Repository Contents

| File | Description |
|---|---|
| `MSCS_634_Lab_3.ipynb` | Complete Jupyter Notebook with code, plots, and written analysis |
| `README.md` | This file — overview, insights, and reflections |

---

## Key Insights

### Performance Results

| Algorithm | Silhouette Score | Adjusted Rand Index (ARI) |
|---|---|---|
| K-Means   | ~0.285 | ~0.898 |
| K-Medoids | ~0.266 | ~0.726 |

> *Exact values are printed when the notebook is executed.*

### K-Means Observations
- Achieved a notably higher ARI (~0.90), indicating strong alignment with the true wine class labels.
- Centroids (computed means) fall near the geometric center of each cluster, enabling accurate boundary decisions.
- Converges efficiently via the Expectation-Maximization (EM) update loop.

### K-Medoids Observations
- ARI (~0.73) is lower because medoids are constrained to actual data points, which may not perfectly anchor cluster centers.
- More robust to outliers by design — an extreme point cannot shift the cluster representative since the medoid must be an observed sample.
- Better suited for non-Euclidean distance metrics (e.g., cosine, Manhattan, or categorical distances).

### When to Prefer Each Algorithm

| Scenario | Preferred |
|---|---|
| Clean, continuous features, large dataset | K-Means |
| Data with significant outliers or noise | K-Medoids |
| Non-Euclidean distance metrics needed | K-Medoids |
| Need interpretable, real-data representatives | K-Medoids |
| Spherical, well-separated clusters | K-Means |
| Medical or financial anomaly-prone data | K-Medoids |

**Bottom line:** On the clean, scaled Wine Dataset with roughly spherical clusters and no significant outliers, K-Means outperforms K-Medoids on both metrics. K-Medoids becomes the better choice when data quality issues, outliers, or distance metric constraints make K-Means unreliable.

---

## Challenges and Decisions

### 1. Z-Score Standardization
The Wine Dataset features span vastly different ranges — `proline` reaches ~1680 while `nonflavanoid_phenols` stays below 1.0. Without `StandardScaler`, distance calculations are dominated by high-magnitude features. Z-score normalization was applied before fitting either model.

### 2. PCA for Visualization
Neither algorithm's clusters can be directly visualized in 13 dimensions. PCA was applied purely for plotting — models were trained on the full 13D scaled feature space. The first two principal components capture roughly 55% of variance and are sufficient to reveal cluster structure visually.

### 3. scikit-learn-extra Compatibility
`KMedoids` is provided by `scikit-learn-extra`, which requires `numpy < 2.0`. Pin `numpy==1.26.4` when installing to avoid import errors in newer environments.

### 4. PAM Method for K-Medoids
The PAM (Partitioning Around Medoids) method was chosen over faster approximations to ensure the most accurate medoid placement. This is feasible given the dataset's small size (178 samples).

### 5. Internal vs. External Metrics
- **Silhouette Score** — an *internal* metric measuring cluster compactness and separation without reference to ground truth (higher is better, max = 1.0).
- **ARI** — an *external* metric comparing discovered clusters to known true labels (1.0 = perfect, 0 = random). ARI is more informative here because ground-truth labels are available.

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/MSCS_634_Lab_3.git
   cd MSCS_634_Lab_3
   ```

2. Install dependencies:
   ```bash
   pip install "numpy==1.26.4" scikit-learn scikit-learn-extra pandas matplotlib notebook
   ```

3. Launch Jupyter:
   ```bash
   jupyter notebook MSCS_634_Lab_3.ipynb
   ```

4. Run all cells via `Kernel > Restart & Run All`.

---

## Requirements

- Python 3.8+
- numpy == 1.26.4
- scikit-learn
- scikit-learn-extra
- pandas
- matplotlib
- jupyter
