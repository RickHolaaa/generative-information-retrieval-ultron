"""
Hierarchical Clustering for Vector ID Generation

This module implements hierarchical clustering as an alternative to Product Quantization (PQ).
Instead of independently quantizing subspaces, it creates a tree structure where each vector's
ID represents its path through the hierarchy.
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from sklearn.cluster import KMeans
import pickle


class HierarchicalQuantizer:
    """
    Hierarchical clustering quantizer that creates a tree of K-means clusters.
    
    Each level of the tree has K clusters, and vectors are assigned IDs based on
    their path through the tree (similar to hierarchical navigable small world graphs).
    
    Args:
        num_levels: Number of levels in the hierarchy (similar to M in PQ)
        num_clusters: Number of clusters at each level (similar to K in PQ)
        random_state: Random seed for reproducibility
        batch_size: Batch size for encoding (default 5000)
    """
    
    def __init__(self, num_levels: int = 4, num_clusters: int = 128, random_state: int = 42, batch_size: int = 5000):
        self.num_levels = num_levels
        self.num_clusters = num_clusters
        self.random_state = random_state
        self.batch_size = batch_size
        
        # Tree structure: list of dicts, where each dict maps parent_path -> KMeans model
        self.tree: List[dict] = [{} for _ in range(num_levels)]
        # Centroids at each level, organized by path
        self.centroids: List[dict] = [{} for _ in range(num_levels)]
        # Flat codewords in format (num_levels, num_clusters, vec_dim) for compatibility
        self.codewords: Optional[np.ndarray] = None
        self.vec_dim: Optional[int] = None
        
    def fit(self, X: np.ndarray, max_points_per_cluster: int = 10000) -> 'HierarchicalQuantizer':
        """
        Fit the hierarchical clustering model.
        
        Args:
            X: Training vectors of shape (n_samples, vec_dim)
            max_points_per_cluster: Maximum points to use for fitting each cluster
        
        Returns:
            self
        """
        self.vec_dim = X.shape[1]
        n_samples = X.shape[0]
        
        print(f"Fitting hierarchical quantizer: {self.num_levels} levels, {self.num_clusters} clusters each")
        
        # Level 0: cluster all data
        print(f"  Level 0: clustering {n_samples} vectors...")
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=self.random_state, n_init=10)
        kmeans.fit(X)
        self.tree[0][()] = kmeans
        self.centroids[0][()] = kmeans.cluster_centers_
        
        # Get assignments for level 0
        assignments = [kmeans.predict(X)]
        
        # Subsequent levels: cluster within each parent cluster
        for level in range(1, self.num_levels):
            print(f"  Level {level}: clustering within {self.num_clusters ** level} parent clusters...")
            
            # For each unique path up to this level
            for parent_idx in range(self.num_clusters ** level):
                # Convert index to path tuple
                parent_path = self._idx_to_path(parent_idx, level)
                
                # Find points belonging to this path
                mask = np.ones(n_samples, dtype=bool)
                for l, cluster_id in enumerate(parent_path):
                    mask &= (assignments[l] == cluster_id)
                
                points_in_cluster = X[mask]
                
                if len(points_in_cluster) < self.num_clusters:
                    # Not enough points, use parent centroid repeated or skip
                    # Create dummy clusters using the points we have
                    if len(points_in_cluster) > 0:
                        # Repeat points to fill clusters
                        fake_centroids = np.tile(
                            points_in_cluster, 
                            (self.num_clusters // max(1, len(points_in_cluster)) + 1, 1)
                        )[:self.num_clusters]
                    else:
                        # Use parent centroid
                        parent_centroid = self._get_centroid(parent_path[:-1], parent_path[-1])
                        fake_centroids = np.tile(parent_centroid, (self.num_clusters, 1))
                    
                    self.centroids[level][parent_path] = fake_centroids
                    self.tree[level][parent_path] = None
                else:
                    # Subsample if too many points
                    if len(points_in_cluster) > max_points_per_cluster:
                        subsample_idx = np.random.choice(
                            len(points_in_cluster), max_points_per_cluster, replace=False
                        )
                        points_for_fit = points_in_cluster[subsample_idx]
                    else:
                        points_for_fit = points_in_cluster
                    
                    kmeans = KMeans(
                        n_clusters=self.num_clusters, 
                        random_state=self.random_state, 
                        n_init=10
                    )
                    kmeans.fit(points_for_fit)
                    self.tree[level][parent_path] = kmeans
                    self.centroids[level][parent_path] = kmeans.cluster_centers_
            
            # Update assignments for this level
            level_assignments = np.zeros(n_samples, dtype=np.int32)
            for i in range(n_samples):
                path = tuple(assignments[l][i] for l in range(level))
                if self.tree[level].get(path) is not None:
                    level_assignments[i] = self.tree[level][path].predict(X[i:i+1])[0]
                else:
                    # Assign to nearest centroid
                    centroids = self.centroids[level].get(path)
                    if centroids is not None:
                        dists = np.linalg.norm(X[i] - centroids, axis=1)
                        level_assignments[i] = np.argmin(dists)
                    else:
                        level_assignments[i] = 0
            
            assignments.append(level_assignments)
        
        # Build flat codewords for compatibility with existing code
        self._build_codewords()
        
        print("Hierarchical quantizer fitting complete!")
        return self
    
    def _idx_to_path(self, idx: int, length: int) -> Tuple[int, ...]:
        """Convert flat index to path tuple."""
        path = []
        for _ in range(length):
            path.append(idx % self.num_clusters)
            idx //= self.num_clusters
        return tuple(reversed(path))
    
    def _get_centroid(self, parent_path: Tuple[int, ...], cluster_id: int) -> np.ndarray:
        """Get centroid for a specific cluster at a path."""
        level = len(parent_path)
        if level == 0:
            return self.centroids[0][()][cluster_id]
        return self.centroids[level][parent_path][cluster_id]
    
    def _build_codewords(self) -> None:
        """Build codewords array in PQ-compatible format (num_levels, num_clusters, vec_dim)."""
        # For hierarchical clustering, we store the level-0 centroids repeated for compatibility
        # Note: This is an approximation since hierarchical centroids depend on the path
        # For a more accurate representation, we average centroids at each level
        
        self.codewords = np.zeros((self.num_levels, self.num_clusters, self.vec_dim), dtype=np.float32)
        
        # Level 0 centroids
        self.codewords[0] = self.centroids[0][()]
        
        # For other levels, average all centroids at that level
        for level in range(1, self.num_levels):
            all_centroids = []
            for path, centroids in self.centroids[level].items():
                all_centroids.append(centroids)
            
            if all_centroids:
                # Stack and compute mean per cluster across all paths
                stacked = np.stack(all_centroids, axis=0)  # (num_paths, num_clusters, vec_dim)
                self.codewords[level] = stacked.mean(axis=0)
    
    def encode(self, X: np.ndarray) -> np.ndarray:
        """
        Encode vectors to hierarchical codes.
        
        Args:
            X: Vectors to encode, shape (n_samples, vec_dim)
        
        Returns:
            Codes of shape (n_samples, num_levels)
        """
        n_samples = X.shape[0]
        codes = np.zeros((n_samples, self.num_levels), dtype=np.int32)
        
        # Process in batches for memory efficiency
        for batch_start in range(0, n_samples, self.batch_size):
            batch_end = min(batch_start + self.batch_size, n_samples)
            X_batch = X[batch_start:batch_end]
            batch_size = batch_end - batch_start
            
            # Level 0
            codes[batch_start:batch_end, 0] = self.tree[0][()].predict(X_batch)
            
            # Subsequent levels
            for level in range(1, self.num_levels):
                for i in range(batch_size):
                    global_idx = batch_start + i
                    parent_path = tuple(codes[global_idx, :level])
                    
                    if self.tree[level].get(parent_path) is not None:
                        codes[global_idx, level] = self.tree[level][parent_path].predict(X_batch[i:i+1])[0]
                    else:
                        # Assign to nearest centroid
                        centroids = self.centroids[level].get(parent_path)
                        if centroids is not None:
                            dists = np.linalg.norm(X_batch[i] - centroids, axis=1)
                            codes[global_idx, level] = np.argmin(dists)
                        else:
                            codes[global_idx, level] = 0
        
        return codes
    
    def decode(self, codes: np.ndarray) -> np.ndarray:
        """
        Decode hierarchical codes to approximate vectors.
        
        Args:
            codes: Codes of shape (n_samples, num_levels)
        
        Returns:
            Reconstructed vectors of shape (n_samples, vec_dim)
        """
        n_samples = codes.shape[0]
        reconstructed = np.zeros((n_samples, self.vec_dim), dtype=np.float32)
        
        for i in range(n_samples):
            # Use the deepest level centroid
            path = tuple(codes[i, :-1])
            cluster_id = codes[i, -1]
            
            centroids = self.centroids[self.num_levels - 1].get(path)
            if centroids is not None:
                reconstructed[i] = centroids[cluster_id]
            else:
                # Fallback to level 0
                reconstructed[i] = self.centroids[0][()][codes[i, 0]]
        
        return reconstructed
    
    def save(self, path: Path) -> None:
        """Save the quantizer to disk."""
        path = Path(path)
        
        save_dict = {
            'num_levels': self.num_levels,
            'num_levels': self.num_levels,
            'num_clusters': self.num_clusters,
            'batch_size': self.batch_size,
            'vec_dim': self.vec_dim,
            'centroids': self.centroids,
            'codewords': self.codewords,
        }
        
        # Save tree separately (KMeans objects)
        tree_dict = {}
        for level in range(self.num_levels):
            tree_dict[level] = {}
            for key, kmeans in self.tree[level].items():
                if kmeans is not None:
                    tree_dict[level][key] = {
                        'cluster_centers_': kmeans.cluster_centers_,
                        'labels_': kmeans.labels_ if hasattr(kmeans, 'labels_') else None,
                    }
        save_dict['tree'] = tree_dict
        
        with open(path / "hierarchical_quantizer.pkl", 'wb') as f:
            pickle.dump(save_dict, f)
        
        # Also save codewords in numpy format for compatibility
        np.save(path / "codebook.npy", self.codewords)
    
    @classmethod
    def load(cls, path: Path, batch_size: int = 5000) -> 'HierarchicalQuantizer':
        """Load a quantizer from disk."""
        path = Path(path)
        
        with open(path / "hierarchical_quantizer.pkl", 'rb') as f:
            save_dict = pickle.load(f)
        
        quantizer = cls(
            num_levels=save_dict['num_levels'],
            num_clusters=save_dict['num_clusters'],
            batch_size=save_dict.get('batch_size', batch_size),
        )
        quantizer.vec_dim = save_dict['vec_dim']
        quantizer.centroids = save_dict['centroids']
        quantizer.codewords = save_dict['codewords']
        
        # Reconstruct tree with minimal KMeans objects
        tree_dict = save_dict['tree']
        for level in range(quantizer.num_levels):
            for key, data in tree_dict[level].items():
                kmeans = KMeans(n_clusters=quantizer.num_clusters, n_init=10)
                kmeans.cluster_centers_ = data['cluster_centers_']
                kmeans._n_features_out = quantizer.vec_dim
                quantizer.tree[level][key] = kmeans
        
        return quantizer


class HierarchicalRetriever(HierarchicalQuantizer):
    """Hierarchical quantizer with retrieval capabilities."""
    
    def get_neighbors(self, q: np.ndarray, codes: np.ndarray, k: int = 100) -> np.ndarray:
        """
        Find k nearest neighbors using asymmetric distance computation.
        
        Args:
            q: Query vector of shape (vec_dim,)
            codes: Database codes of shape (n_samples, num_levels)
            k: Number of neighbors to return
        
        Returns:
            Indices of k nearest neighbors
        """
        # Decode all vectors and compute distances
        reconstructed = self.decode(codes)
        dists = np.linalg.norm(reconstructed - q, axis=1)
        return np.argsort(dists)[:k]


if __name__ == "__main__":
    # Test the hierarchical quantizer
    np.random.seed(42)
    
    # Generate random test data
    n_samples = 1000
    vec_dim = 128
    X = np.random.randn(n_samples, vec_dim).astype(np.float32)
    
    # Fit quantizer
    quantizer = HierarchicalQuantizer(num_levels=4, num_clusters=16)
    quantizer.fit(X)
    
    # Encode
    codes = quantizer.encode(X)
    print(f"Codes shape: {codes.shape}")
    print(f"Sample codes:\n{codes[:5]}")
    
    # Decode
    reconstructed = quantizer.decode(codes)
    print(f"Reconstructed shape: {reconstructed.shape}")
    
    # Compute reconstruction error
    mse = np.mean((X - reconstructed) ** 2)
    print(f"Mean squared error: {mse:.4f}")
    
    # Test save/load
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        quantizer.save(tmpdir)
        loaded = HierarchicalQuantizer.load(tmpdir)
        codes2 = loaded.encode(X[:10])
        print(f"Codes match after save/load: {np.allclose(codes[:10], codes2)}")
