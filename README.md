```md
# Gradient-Based Weight Clustering (GBWC)

This repository contains the implementation of the Gradient-Based Weight Clustering (GBWC) algorithm. The algorithm is designed to learn feature weights automatically during the clustering process to handle high-dimensional data effectively.

## Dependencies

The following libraries are required to run the project:

```bash
pip install numpy pandas scikit-learn scipy tqdm matplotlib
```

## Project Structure

*   `gbwc_class.py`: Implementation of the core GBWC clustering algorithm.
*   `optimizer_gbwc.py`: Parameter optimizer for finding optimal lambda and beta values.
*   `metric_fusion_weights.csv`: Configuration file for metric fusion weights (optional).
*   `run.py`: Entry point for data loading, clustering, and evaluation.

## Usage

You can execute the clustering pipeline directly by running the `run.py` script. This script performs data preprocessing, fits the GBWC model, calculates multiple evaluation metrics (NMI, ARI, etc.), and visualizes the resulting feature weights.

```bash
python run.py
```

## Parameter Optimization

The project includes an optimization class (`GBWCParaOptimizer`) that performs a grid search based on unsupervised metrics such as Silhouette Score, Calinski-Harabasz Index, and Davies-Bouldin Index. The scoring logic utilizes the `metric_fusion_weights.csv` file if available; otherwise, it defaults to a uniform weighting scheme.

```
