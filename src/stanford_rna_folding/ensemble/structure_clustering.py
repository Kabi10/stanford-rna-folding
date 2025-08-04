"""
Structure clustering for RNA ensemble prediction.

This module provides tools to cluster similar predicted structures and select 
representative structures for each cluster to ensure diversity in the ensemble.
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Union, Any
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist, squareform

from ..evaluation.metrics import batch_rmsd, batch_tm_score


class StructureClusterer:
    """
    Cluster similar predicted RNA structures and select representatives.
    """
    
    def __init__(
        self, 
        clustering_method: str = "kmeans",
        n_clusters: int = 5,
        distance_metric: str = "rmsd",
        min_cluster_size: int = 2,
        eps: float = 2.0,  # For DBSCAN
        max_structures_per_cluster: int = 3,
        device: str = "cpu",
    ):
        """
        Initialize the structure clusterer.
        
        Args:
            clustering_method: Clustering algorithm ('kmeans', 'dbscan', 'agglomerative')
            n_clusters: Number of clusters for KMeans or AgglomerativeClustering
            distance_metric: Distance metric for clustering ('rmsd', 'tm_score', 'coordinate')
            min_cluster_size: Minimum number of structures in a cluster (for DBSCAN)
            eps: Maximum distance between samples for DBSCAN
            max_structures_per_cluster: Maximum number of structures to select per cluster
            device: Computation device ('cpu' or 'cuda')
        """
        self.clustering_method = clustering_method
        self.n_clusters = n_clusters
        self.distance_metric = distance_metric
        self.min_cluster_size = min_cluster_size
        self.eps = eps
        self.max_structures_per_cluster = max_structures_per_cluster
        self.device = device
        
        self.structures = None
        self.distance_matrix = None
        self.labels = None
        self.representative_indices = None
    
    def fit(self, structures: torch.Tensor) -> np.ndarray:
        """
        Fit the clusterer to a set of predicted structures.
        
        Args:
            structures: Tensor of shape (num_structures, seq_len, num_atoms, 3) containing
                        multiple predicted structures for the same RNA sequence.
                        
        Returns:
            Array of cluster labels for each structure
        """
        self.structures = structures
        
        # Compute pairwise distance matrix
        distance_matrix = self._compute_distance_matrix(structures)
        self.distance_matrix = distance_matrix
        
        # Apply clustering algorithm
        if self.clustering_method == "kmeans":
            # For KMeans, we need to convert structures to a 2D feature matrix
            # We flatten each structure to a feature vector
            features = structures.reshape(structures.shape[0], -1).cpu().numpy()
            clusterer = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            self.labels = clusterer.fit_predict(features)
            
        elif self.clustering_method == "dbscan":
            # DBSCAN works directly on the distance matrix
            clusterer = DBSCAN(
                eps=self.eps, 
                min_samples=self.min_cluster_size,
                metric="precomputed"
            )
            self.labels = clusterer.fit_predict(distance_matrix)
            
        elif self.clustering_method == "agglomerative":
            # Agglomerative clustering with precomputed distances
            clusterer = AgglomerativeClustering(
                n_clusters=self.n_clusters,
                affinity="precomputed",
                linkage="average"
            )
            self.labels = clusterer.fit_predict(distance_matrix)
        
        else:
            raise ValueError(f"Unsupported clustering method: {self.clustering_method}")
        
        return self.labels
    
    def select_representatives(self, quality_scores: Optional[np.ndarray] = None) -> List[int]:
        """
        Select representative structures from each cluster.
        
        Args:
            quality_scores: Optional quality scores for each structure (higher is better).
                           If not provided, structures closest to cluster centers are selected.
                           
        Returns:
            List of indices of selected representative structures
        """
        if self.labels is None:
            raise ValueError("Clusterer must be fitted before selecting representatives")
        
        representatives = []
        unique_labels = np.unique(self.labels)
        
        # Skip noise points (label -1) if using DBSCAN
        if -1 in unique_labels and self.clustering_method == "dbscan":
            unique_labels = unique_labels[unique_labels != -1]
        
        for label in unique_labels:
            cluster_indices = np.where(self.labels == label)[0]
            
            if len(cluster_indices) == 0:
                continue
                
            # Select representatives based on quality or distance to center
            if quality_scores is not None:
                # Select top structures by quality score
                cluster_scores = quality_scores[cluster_indices]
                sorted_indices = np.argsort(-cluster_scores)  # Sort in descending order
                selected = [cluster_indices[i] for i in sorted_indices[:self.max_structures_per_cluster]]
            else:
                # Select structures closest to cluster center
                selected = self._select_by_centrality(cluster_indices)
            
            representatives.extend(selected)
        
        self.representative_indices = representatives
        return representatives
    
    def _select_by_centrality(self, cluster_indices: np.ndarray) -> List[int]:
        """
        Select structures closest to the cluster center.
        
        Args:
            cluster_indices: Indices of structures in the cluster
            
        Returns:
            List of indices of selected representative structures
        """
        if len(cluster_indices) <= self.max_structures_per_cluster:
            return cluster_indices.tolist()
        
        # Extract the distance sub-matrix for this cluster
        cluster_distances = self.distance_matrix[np.ix_(cluster_indices, cluster_indices)]
        
        # Compute centrality as the sum of distances to all other points
        centrality = np.sum(cluster_distances, axis=1)
        
        # Select the most central structures (lowest sum of distances)
        central_indices = np.argsort(centrality)[:self.max_structures_per_cluster]
        
        # Map back to original indices
        return [cluster_indices[i] for i in central_indices]
    
    def _compute_distance_matrix(self, structures: torch.Tensor) -> np.ndarray:
        """
        Compute pairwise distance matrix between structures.
        
        Args:
            structures: Tensor of shape (num_structures, seq_len, num_atoms, 3)
            
        Returns:
            Distance matrix of shape (num_structures, num_structures)
        """
        num_structures = structures.shape[0]
        distance_matrix = np.zeros((num_structures, num_structures))
        
        # Move to device
        structures = structures.to(self.device)
        
        for i in range(num_structures):
            for j in range(i+1, num_structures):
                if self.distance_metric == "rmsd":
                    # Compute RMSD between structures i and j
                    rmsd = batch_rmsd(
                        structures[i:i+1],  # Add batch dimension
                        structures[j:j+1]   # Add batch dimension
                    ).item()
                    distance = rmsd
                    
                elif self.distance_metric == "tm_score":
                    # Compute 1 - TM_score (since TM score is a similarity measure)
                    tm_score = batch_tm_score(
                        structures[i:i+1],
                        structures[j:j+1]
                    ).item()
                    distance = 1.0 - tm_score
                    
                elif self.distance_metric == "coordinate":
                    # Euclidean distance in flattened coordinate space
                    struct_i = structures[i].reshape(-1)
                    struct_j = structures[j].reshape(-1)
                    distance = torch.norm(struct_i - struct_j).item()
                
                else:
                    raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
                
                # Symmetric matrix
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
        
        return distance_matrix
    
    def visualize_clusters(self, output_file: Optional[str] = None) -> None:
        """
        Visualize the clusters using t-SNE for dimensionality reduction.
        
        Args:
            output_file: Optional path to save the visualization
        """
        if self.structures is None or self.labels is None:
            raise ValueError("Clusterer must be fitted before visualization")
        
        # Flatten structures for t-SNE
        flattened = self.structures.reshape(self.structures.shape[0], -1).cpu().numpy()
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42, metric="euclidean")
        embedded = tsne.fit_transform(flattened)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        # Plot all points colored by cluster
        unique_labels = np.unique(self.labels)
        for label in unique_labels:
            if label == -1:
                # Plot noise points in black
                noise_mask = self.labels == -1
                plt.scatter(
                    embedded[noise_mask, 0], 
                    embedded[noise_mask, 1], 
                    color="black", 
                    marker="x", 
                    label="Noise"
                )
            else:
                # Plot clusters with different colors
                cluster_mask = self.labels == label
                plt.scatter(
                    embedded[cluster_mask, 0], 
                    embedded[cluster_mask, 1], 
                    label=f"Cluster {label}"
                )
        
        # Highlight representative structures if selected
        if self.representative_indices is not None:
            plt.scatter(
                embedded[self.representative_indices, 0],
                embedded[self.representative_indices, 1],
                s=100, 
                edgecolor="black", 
                facecolor="none", 
                linewidth=2, 
                label="Representatives"
            )
        
        plt.title(f"Structure Clusters ({self.clustering_method})")
        plt.legend()
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300)
            plt.close()
        else:
            plt.show()


def cluster_structures(
    structures: torch.Tensor,
    n_clusters: int = 5,
    method: str = "kmeans",
    distance_metric: str = "rmsd",
    quality_scores: Optional[np.ndarray] = None,
    visualize: bool = False,
    output_dir: Optional[str] = None,
    device: str = "cpu",
) -> Tuple[np.ndarray, List[int]]:
    """
    Cluster RNA structures and select representatives from each cluster.
    
    Args:
        structures: Tensor of shape (num_structures, seq_len, num_atoms, 3)
        n_clusters: Number of clusters to form
        method: Clustering method ('kmeans', 'dbscan', 'agglomerative')
        distance_metric: Metric for comparing structures ('rmsd', 'tm_score', 'coordinate')
        quality_scores: Optional quality scores for each structure (higher is better)
        visualize: Whether to generate visualization
        output_dir: Directory to save visualization if generated
        device: Computation device ('cpu' or 'cuda')
        
    Returns:
        Tuple of (cluster labels, representative indices)
    """
    clusterer = StructureClusterer(
        clustering_method=method,
        n_clusters=n_clusters,
        distance_metric=distance_metric,
        device=device
    )
    
    # Fit clusterer to structures
    labels = clusterer.fit(structures)
    
    # Select representative structures
    representatives = clusterer.select_representatives(quality_scores)
    
    # Visualize if requested
    if visualize and output_dir:
        output_path = Path(output_dir) / f"structure_clusters_{method}.png"
        clusterer.visualize_clusters(str(output_path))
    
    return labels, representatives 