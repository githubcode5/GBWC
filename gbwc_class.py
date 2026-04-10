# -*- coding: utf-8 -*-

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans
from sklearn.utils.validation import check_array, check_is_fitted
from scipy.sparse.linalg import svds
import gc

EPS = 1e-9

class GBWC(BaseEstimator, ClusterMixin):
    """
    Gradient-Based Weight Clustering (GBWC)
    
    Parameters:
    -----------
    n_clusters : int
        The number of clusters to form.
    lambda_val : float, default=0.02
        Sparsity regularization parameter.
    beta : float, default=3.0
        Power parameter for weight updating.
    max_iter : int, default=30
        Maximum number of iterations.
    random_state : int, default=7
        Random seed for reproducibility.
    """
    def __init__(self, n_clusters, lambda_val=2, beta=3.0, 
                 max_iter=30, random_state=7):
        self.n_clusters = n_clusters
        self.lambda_val = lambda_val
        self.beta = beta
        self.max_iter = max_iter
        self.random_state = random_state
    
    def fit(self, X, y=None):
        # [Optimization] Force float32 to prevent memory doubling caused by 64-bit conversion
        X = check_array(X, accept_sparse=False, dtype=np.float32)
        
        labels, centers, weights, n_iter = self._gbwc_algorithm(X)
        
        self.labels_ = labels
        self.cluster_centers_ = centers
        self.weights_ = weights
        self.n_iter_ = n_iter
        return self
    
    def fit_predict(self, X, y=None):
        return self.fit(X).labels_
    
    def predict(self, X):
        check_is_fitted(self, ['cluster_centers_', 'weights_'])
        # [Optimization] Force float32 during prediction phase
        X = check_array(X, accept_sparse=False, dtype=np.float32)
        
        n_samples = X.shape[0]
        n_clusters = self.cluster_centers_.shape[0]
        distances = np.zeros((n_samples, n_clusters), dtype=np.float32)
        
        for i in range(n_clusters):
            # In-place subtraction to reduce intermediate memory copies
            diff = X - self.cluster_centers_[i]
            distances[:, i] = np.sum(diff ** 2 * self.weights_, axis=1)
        
        return np.argmin(distances, axis=1)

    def _gbwc_algorithm(self, X):
        if self.n_clusters > len(X):
            raise ValueError(f"Number of clusters (k) cannot be greater than number of samples.")
        
        rng = np.random.RandomState(self.random_state)
        
        # Warm start
        X, w, alpha, lambda_scaled = self._warm_start(X)

        n, p = X.shape
        w = (w / w.sum()).astype(np.float32)
        
        # Build initial balls
        balls = self._build_balls(X, w)
        B = len(balls)
        
        C_orig_all = np.vstack([b['center'] for b in balls]).astype(np.float32)
        N = np.array([b['n'] for b in balls], dtype=np.float32)
        
        km = KMeans(n_clusters=self.n_clusters, n_init=8, random_state=self.random_state)
        ball_labels = km.fit_predict(C_orig_all * w, sample_weight=N)
        
        M = np.zeros((self.n_clusters, p), dtype=np.float32)
        # Initialize cluster centers
        for j in range(self.n_clusters):
            idx = np.where(ball_labels == j)[0]
            if len(idx) == 0:
                M[j] = C_orig_all[rng.randint(B)]
            else:
                M[j] = (C_orig_all[idx] * N[idx, None]).sum(axis=0) / N[idx].sum()
        
        M_prev = np.zeros((self.n_clusters, p), dtype=np.float32)
        w_prev = np.zeros(p, dtype=np.float32)
        
        for t in range(self.max_iter):
            # 1. Update Cluster Centers (M)
            for j in range(self.n_clusters):
                idx = np.where(ball_labels == j)[0]
                if len(idx) == 0:
                    M[j] = C_orig_all[rng.randint(B)]
                else:
                    M[j] = (C_orig_all[idx] * N[idx, None]).sum(axis=0) / N[idx].sum()
            
            # 2. Compute Dispersion (D)
            D = self._compute_D_from_balls(balls, ball_labels, M, p)
            
            # 3. Update feature weights
            w, _ = self._update_weights(D, alpha, lambda_scaled)
            w = (w / w.sum()).astype(np.float32)
            
            center_shift = np.linalg.norm(M - M_prev)
            M_prev = M.copy()
            
            # Rebuild balls only when centers shift significantly to save time and memory
            if (center_shift > 1e-5) or t < 5:
                # Dist_matrix scale is limited by number of balls (B), which is usually small
                dist_matrix = np.linalg.norm(C_orig_all[:, None, :] - M[None, :, :], axis=2)
                d_sorted = np.sort(dist_matrix, axis=1)
                
                rel_margin = (d_sorted[:, 1] - d_sorted[:, 0]) / (d_sorted[:, 0] + EPS)
                mg_med = np.median(rel_margin)
                mg_iqr = np.percentile(rel_margin, 75) - np.percentile(rel_margin, 25)
                score_margin = np.clip((mg_med + mg_iqr - rel_margin) / (mg_iqr + EPS), 0, 1)
                
                R_all = np.array([b['radius'] for b in balls], dtype=np.float32)
                valid_w_count = len(list(filter(lambda x: x > 1e-4, w)))
                density = N / (R_all ** valid_w_count + EPS)
                
                den_med = np.median(density)
                den_iqr = np.percentile(density, 75) - np.percentile(density, 25)
                score_sparse = np.clip((den_med - density) / (den_iqr + EPS), 0, 1)
                
                progress = t / self.max_iter
                w_margin = 0.1 + 0.8 * progress  
                pos = max(0.2, 0.9 - 1.2 * progress)
                edge_score = (w_margin * score_margin + (1 - w_margin) * score_sparse)
                
                th = np.percentile(edge_score, 100 * (1 - pos))
                condition = (edge_score >= th) & (N > 2)
                edge_idx = np.where(condition)[0]
                
                if len(edge_idx) > 0:
                    balls = self._rebuild_edge_balls(balls, edge_idx, X, w)
                    # Manually trigger garbage collection after rebuilding to free memory
                    gc.collect() 
                    C_orig_all = np.vstack([b['center'] for b in balls]).astype(np.float32)
                    N = np.array([b['n'] for b in balls], dtype=np.float32)
                    B = len(balls)
            
            # Update ball labels
            new_labels = np.zeros(B, dtype=int)
            for i in range(B):
                # Use float32 differences explicitly
                diff2 = (C_orig_all[i:i + 1] - M) ** 2
                d = np.sum(diff2 * w, axis=1)
                new_labels[i] = np.argmin(d)
            ball_labels = new_labels
            
            if (center_shift < 1e-4 and np.linalg.norm(w - w_prev) < 1e-4 and t > 5):
                break
            w_prev = w.copy()
        
        sample_labels = np.zeros(n, dtype=np.int32)
        for bi, b in enumerate(balls):
            sample_labels[b['members']] = ball_labels[bi]
        
        return sample_labels, M, w, t + 1
    
    def _compute_D_from_balls(self, balls, ball_labels, M, p):
        D = np.zeros(p, dtype=np.float32)
        for i, b in enumerate(balls):
            lab = ball_labels[i]
            # Accumulate in-place to avoid creating temporary copies
            D += b['internal_dist']
            D += b['n'] * ((b['center'] - M[lab]) ** 2)
        return D
    
    def _update_weights(self, D, alpha, lambda_scaled):
        # Force float32 for weighting calculations
        D_safe = D.astype(np.float32) + EPS
        D_inv = 1.0 / D_safe
        D_pow = D_inv ** (1 / (self.beta - 1))
        sum_D = np.sum(D_pow)
        alpha = (1.0 / (sum_D + EPS)) ** (self.beta - 1)
        
        p = len(D)
        w = np.zeros(p, dtype=np.float32)
        cond = alpha > lambda_scaled * D_safe
        if np.any(cond):
            val = (alpha / D_safe[cond]) - lambda_scaled
            val = np.maximum(val, 0.0)
            w[cond] = val ** (1.0 / (self.beta - 1.0))
        
        s = w.sum()
        return w if s > 0 else np.ones(p, dtype=np.float32) / p, s
    
    def _warm_start(self, X):
        # Switch initialization calculations to float32
        centroid = np.mean(X, axis=0, dtype=np.float32)
        D = np.mean((X - centroid) ** 2, axis=0, dtype=np.float32)
        
        # Remove features with near-zero variance
        zero_indices = np.where(D < EPS)[0]
        if len(zero_indices) > 0:
            D = np.delete(D, zero_indices)
            X = np.delete(X, zero_indices, axis=1)
        
        D = np.maximum(D, 1e-10)
        n, p = X.shape
        lambda_scaled = self.lambda_val / (p * p * 100)
        
        # Initial weight update
        D_inv = 1 / D
        pw = D_inv ** (1 / (self.beta - 1))
        S = pw.sum()
        alpha = (1 / (S + 1e-10)) ** (self.beta - 1)
        
        w, s = self._update_weights(D, alpha, lambda_scaled)
        return X, w, alpha, lambda_scaled
    
    def _build_balls(self, X, w):
        n = X.shape[0]
        balls = [{
            'center': np.mean(X, axis=0, dtype=np.float32),
            'internal_dist': np.sum((X - np.mean(X, axis=0))**2, axis=0, dtype=np.float32),
            'radius': 1.0, # Initial placeholder
            'members': np.arange(n),
            'n': n
        }]
        
        i = 0
        while len(balls) < self.n_clusters:
            N_counts = np.array([b['n'] for b in balls])
            threshold = np.percentile(N_counts, 50)
            edge_idx = np.where(N_counts >= threshold)[0]
            balls = self._rebuild_edge_balls(balls, edge_idx, X, w)
            i += 1
            if i > 15: break
        return balls
    
    def _rebuild_edge_balls(self, balls, edge_idx, X, w):
        """[Core Optimization] Splitting logic to minimize memory allocation"""
        new_balls = []
        edge_idx_set = set(edge_idx)
        
        for i in edge_idx:
            ball = balls[i]
            members_prev = ball['members']
            if len(members_prev) < 2:
                new_balls.append(ball)
                continue
            
            # Note: X[members_prev] creates a view/copy. 
            # Ensure X is float32 to keep copy size halved compared to float64.
            # Explicitly multiply by weights without extra intermediate objects.
            Xw_sub = X[members_prev] * w.astype(np.float32)
            
            center_w = np.mean(Xw_sub, axis=0, dtype=np.float32)
            Xw_sub -= center_w  # Subtract mean in-place
            
            try:
                # Use SVD to find the direction of maximum variance for splitting
                _, _, Vt = svds(Xw_sub, k=1, which='LM')
                direction = Vt[0]
                labels = (Xw_sub @ direction) > 0
            except:
                labels = np.random.rand(len(members_prev)) > 0.5
            
            # Clean up memory used by weighted sub-matrix
            del Xw_sub
            
            for label_val in [False, True]:
                sub_mask = (labels == label_val)
                if not np.any(sub_mask): continue
                
                new_members = members_prev[sub_mask]
                X_sub_orig = X[new_members] # Creates a sub-copy
                C_orig = np.mean(X_sub_orig, axis=0, dtype=np.float32)
                
                # Directly accumulate internal distance instead of storing differences
                internal_dist = np.sum((X_sub_orig - C_orig)**2, axis=0, dtype=np.float32)
                
                # Calculate weighted radius
                diff_w = (X_sub_orig * w) - (C_orig * w)
                R_w = np.sqrt(np.max(np.sum(diff_w**2, axis=1)))
                
                new_balls.append({
                    'center': C_orig,
                    'internal_dist': internal_dist,
                    'radius': R_w,
                    'members': new_members,
                    'n': len(new_members)
                })
                del X_sub_orig # Delete copy immediately
            
        kept_balls = [b for i, b in enumerate(balls) if i not in edge_idx_set]
        kept_balls.extend(new_balls)
        return kept_balls