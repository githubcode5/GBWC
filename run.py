import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from gbwc_class import GBWC
# from Opt import GBWCParaOptimizer


from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
    fowlkes_mallows_score
)

# Load data
loaded = np.load("data/synthetic_data1.npz")
X = loaded["X"]
y_true = loaded["Class_true"]
n_clusters = len(np.unique(y_true))

# Data preprocessing
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# optimizer = GBWCParaOptimizer(n_clusters=n_clusters, weights_path='metric_fusion_weights.csv')
# best_params = optimizer.optimize(X)

# Model training and prediction
gbwc = GBWC(n_clusters=n_clusters, lambda_val=1, beta=4)
label = gbwc.fit_predict(X)

# Calculate and print all evaluation metrics
print("="*10 + " GBWC Clustering Performance " + "="*10)
nmi = normalized_mutual_info_score(y_true, label)
ari = adjusted_rand_score(y_true, label)
ami = adjusted_mutual_info_score(y_true, label)
homo = homogeneity_score(y_true, label)
comp = completeness_score(y_true, label)
v_m = v_measure_score(y_true, label)
fmi = fowlkes_mallows_score(y_true, label)

print(f"Normalized Mutual Information (NMI) : {nmi:.3f}")
print(f"Adjusted Rand Index (ARI)          : {ari:.3f}")
print(f"Adjusted Mutual Information (AMI)  : {ami:.3f}")
print(f"Homogeneity                        : {homo:.3f}")
print(f"Completeness                       : {comp:.3f}")
print(f"V-Measure                          : {v_m:.3f}")
print(f"Fowlkes-Mallows Score (FMI)        : {fmi:.3f}")
print("="*49)

# Plot feature weights
weights = gbwc.weights_
n_features = X.shape[1]
plt.figure(figsize=(10, 6))
plt.bar(range(n_features), weights, color='royalblue', edgecolor='black', alpha=0.8)
plt.title('Feature Weights Learned by GBWC', fontsize=14, fontweight='bold')
plt.xlabel('Feature Index', fontsize=12)
plt.ylabel('Weight Value', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()