import numpy as np
import pandas as pd
import csv
import os
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from gbwc_class import GBWC

class GBWCParaOptimizer:
    """
    Parameter Optimizer for Gradient-Based Weight Clustering (GBWC).
    
    This class performs a grid search to find the optimal (lambda, beta) 
    combination by fusing multiple unsupervised clustering metrics 
    (SC, CH, DB, Loss) using a pre-defined weighting scheme.
    """
    def __init__(self, n_clusters, weights_path='metric_fusion_weights.csv', random_state=7):
        """
        Initialize the optimizer.
        
        :param n_clusters: The number of target clusters.
        :param weights_path: Path to the TSV file containing metric weights.
        :param random_state: Seed for reproducibility.
        """
        self.n_clusters = n_clusters
        self.weights_path = weights_path
        self.random_state = random_state
        
        # Load weights from file or use fallback defaults
        self.fusion_weights = self._load_fusion_weights()
        
        self.best_params_ = None
        self.results_df_ = None

    def _load_fusion_weights(self):
        """
        Loads weights from the specified CSV/TSV file.
        If the file is not found, it falls back to a uniform weighting scheme.
        """
        weights = {}
        if not os.path.exists(self.weights_path):
            print(f"Warning: Weight file '{self.weights_path}' not found.")
            print("Action: Falling back to DEFAULT UNIFORM WEIGHTS (SC, CH, DB, loss).")
            # Default weights: Higher is better for SC/CH, Lower is better for DB/Loss
            return {
                'SC': 1.0, 
                'CH': 1.0, 
                'DB': -1.0, 
                'loss': -1.0 
            }
            
        try:
            with open(self.weights_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='\t')
                for row in reader:
                    if len(row) != 2: continue
                    key, val_str = row
                    try:
                        weights[key] = float(val_str)
                    except ValueError: continue
            return weights
        except Exception as e:
            print(f"Error reading weight file: {e}. Using empty weights.")
            return {}

    def _calculate_loss(self, X, labels, centers, weights):
        """
        Calculates the weighted within-cluster dispersion (Clustering Loss).
        Consistent with the GBWC objective function logic.
        """
        total_loss = 0
        for i in range(self.n_clusters):
            mask = (labels == i)
            if np.any(mask):
                diff = X[mask] - centers[i]
                total_loss += np.sum((diff**2) * weights)
        return total_loss

    def optimize(self, X, lambda_grid=None, beta_grid=None):
        """
        Executes grid search with a progress bar and returns the best parameters.
        
        :param X: Preprocessed feature matrix (scaled).
        :param lambda_grid: List of lambda values to search.
        :param beta_grid: List of beta values to search.
        :return: Dict containing the best 'lambda' and 'beta'.
        """
        # Set default grids if none provided
        l_grid = lambda_grid if lambda_grid is not None else [0.1, 0.2, 0.5, 1, 2]
        b_grid = beta_grid if beta_grid is not None else [2, 3, 4, 5, 6, 7]
        
        results = []
        total_steps = len(l_grid) * len(b_grid)
        
        # Grid Search loop with Progress Bar
        with tqdm(total=total_steps, desc="Optimizing GBWC Parameters") as pbar:
            for l in l_grid:
                for b in b_grid:
                    model = GBWC(n_clusters=self.n_clusters, lambda_val=l, beta=b, 
                                 random_state=self.random_state)
                    labels = model.fit_predict(X)
                    
                    # Ensure clustering produced valid labels (not all same class)
                    if len(np.unique(labels)) >= 2:
                        sc = silhouette_score(X, labels)
                        ch = calinski_harabasz_score(X, labels)
                        db = davies_bouldin_score(X, labels)
                        loss = self._calculate_loss(X, labels, model.cluster_centers_, model.weights_)
                        
                        results.append({
                            'lambda': l,
                            'beta': b,
                            'SC': sc,
                            'CH': ch,
                            'DB': db,
                            'loss': loss
                        })
                    pbar.update(1)

        if not results:
            raise ValueError("Grid search failed to produce any valid clustering results.")

        # --- Scoring Logic ---
        df_raw = pd.DataFrame(results)
        df_work = df_raw.copy()
        
        # 1. Local Normalization (0-1) to make metrics comparable
        scaler = MinMaxScaler()
        metrics_to_norm = ['SC', 'CH', 'DB', 'loss']
        df_work[metrics_to_norm] = scaler.fit_transform(df_raw[metrics_to_norm])
        
        # 2. Construct Interaction Terms (Collaborative synergy)
        df_work['SC_CH']   = df_work['SC'] * df_work['CH']
        df_work['SC_loss'] = df_work['SC'] * df_work['loss']
        df_work['DB_loss'] = df_work['DB'] * df_work['loss']
        df_work['SC_DB']   = df_work['SC'] * df_work['DB']
        df_work['CH_loss'] = df_work['CH'] * df_work['loss']
        df_work['CH_DB']   = df_work['CH'] * df_work['DB']

        # 3. Calculate Final Weighted Fusion Score
        df_raw['Final_Score'] = 0
        for feat, weight in self.fusion_weights.items():
            if feat in df_work.columns:
                df_raw['Final_Score'] += df_work[feat] * weight
        
        # Final Fallback: If score is still zero, use Silhouette Score as default
        if (df_raw['Final_Score'] == 0).all():
            print("Notice: No valid weights matched. Defaulting to SC score.")
            df_raw['Final_Score'] = df_work['SC']

        # 4. Extract Best Parameters
        best_idx = df_raw['Final_Score'].idxmax()
        self.best_params_ = {
            'lambda': df_raw.loc[best_idx, 'lambda'],
            'beta': df_raw.loc[best_idx, 'beta']
        }
        self.results_df_ = df_raw
        
        print(f"\nOptimization complete. Best Params: {self.best_params_}")
        return self.best_params_