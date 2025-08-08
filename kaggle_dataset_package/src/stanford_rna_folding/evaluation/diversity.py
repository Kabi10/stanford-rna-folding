"""
Structural diversity evaluation for RNA 3D structure predictions.

This module provides tools for measuring structural diversity between multiple 
RNA structure predictions, clustering similar structures, and selecting diverse
representative structures.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans, DBSCAN

# Import calculate_tm_score from alignment module
try:
    from .alignment import calculate_tm_score
except ImportError:
    # Fallback implementation if alignment module is not available
    def calculate_tm_score(struct1, struct2):
        # Simple placeholder implementation
        if isinstance(struct1, torch.Tensor):
            struct1 = struct1.detach().cpu().numpy()
        if isinstance(struct2, torch.Tensor):
            struct2 = struct2.detach().cpu().numpy()
        
        diff = np.mean((struct1 - struct2)**2)
        return 1.0 / (1.0 + diff)


class StructuralDiversityEvaluator:
    """
    Evaluator for measuring structural diversity between multiple RNA predictions.
    """
    
    def __init__(
        self,
        clustering_method: str = "kmeans",  # Options: kmeans, dbscan, hierarchical
        distance_metric: str = "rmsd",      # Options: rmsd, dme, lddt, tm_score
        cluster_threshold: float = 0.3,
    ):
        """
        Initialize the structural diversity evaluator.
        
        Args:
            clustering_method: Method for clustering similar structures
            distance_metric: Metric for measuring distance between structures
            cluster_threshold: Threshold for clustering (interpretation depends on method)
        """
        self.clustering_method = clustering_method
        self.distance_metric = distance_metric
        self.cluster_threshold = cluster_threshold
    
    def _convert_structures_to_numpy(
        self, 
        structures: List[Union[torch.Tensor, np.ndarray]]
    ) -> List[np.ndarray]:
        """
        Convert a list of structures to numpy arrays.
        
        Args:
            structures: List of structures, each of shape (seq_len, num_atoms, 3)
            
        Returns:
            List of numpy array structures
        """
        numpy_structures = []
        
        for struct in structures:
            if isinstance(struct, torch.Tensor):
                struct = struct.detach().cpu().numpy()
            numpy_structures.append(struct)
        
        return numpy_structures
    
    def calculate_rmsd(
        self, 
        struct1: np.ndarray,
        struct2: np.ndarray,
        atom_indices: Optional[List[int]] = None
    ) -> float:
        """
        Calculate the Root Mean Square Deviation (RMSD) between two structures.
        
        Lower RMSD means more similar structures.
        
        Args:
            struct1: First structure of shape (seq_len, num_atoms, 3)
            struct2: Second structure of shape (seq_len, num_atoms, 3)
            atom_indices: Optional indices of atoms to consider (e.g., only backbone)
            
        Returns:
            RMSD between the structures
        """
        if struct1.shape != struct2.shape:
            raise ValueError(f"Structures have different shapes: {struct1.shape} vs {struct2.shape}")
        
        # Select atoms if specified
        if atom_indices is not None:
            struct1 = struct1[:, atom_indices, :]
            struct2 = struct2[:, atom_indices, :]
        
        # Calculate squared differences
        squared_diff = np.sum((struct1 - struct2) ** 2, axis=(1, 2))
        
        # Calculate RMSD
        rmsd = np.sqrt(np.mean(squared_diff))
        
        return rmsd
    
    def calculate_dme(
        self, 
        struct1: np.ndarray,
        struct2: np.ndarray,
        atom_indices: Optional[List[int]] = None
    ) -> float:
        """
        Calculate the Distance Matrix Error (DME) between two structures.
        
        Measures the difference in internal distances between atoms.
        
        Args:
            struct1: First structure of shape (seq_len, num_atoms, 3)
            struct2: Second structure of shape (seq_len, num_atoms, 3)
            atom_indices: Optional indices of atoms to consider
            
        Returns:
            DME between the structures
        """
        if struct1.shape != struct2.shape:
            raise ValueError(f"Structures have different shapes: {struct1.shape} vs {struct2.shape}")
        
        # Select atoms if specified
        if atom_indices is not None:
            struct1 = struct1[:, atom_indices, :]
            struct2 = struct2[:, atom_indices, :]
        
        # Reshape to (seq_len * num_atoms, 3)
        struct1_flat = struct1.reshape(-1, 3)
        struct2_flat = struct2.reshape(-1, 3)
        
        # Calculate distance matrices
        n_atoms = struct1_flat.shape[0]
        dm1 = np.zeros((n_atoms, n_atoms))
        dm2 = np.zeros((n_atoms, n_atoms))
        
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                dm1[i, j] = np.sqrt(np.sum((struct1_flat[i] - struct1_flat[j]) ** 2))
                dm2[i, j] = np.sqrt(np.sum((struct2_flat[i] - struct2_flat[j]) ** 2))
                
                # Make the matrices symmetric
                dm1[j, i] = dm1[i, j]
                dm2[j, i] = dm2[i, j]
        
        # Calculate DME
        dme = np.sqrt(np.mean((dm1 - dm2) ** 2))
        
        return dme
    
    def calculate_lddt(
        self, 
        struct1: np.ndarray,
        struct2: np.ndarray,
        atom_indices: Optional[List[int]] = None,
        cutoffs: List[float] = [0.5, 1.0, 2.0, 4.0]
    ) -> float:
        """
        Calculate the Local Distance Difference Test (lDDT) score between two structures.
        
        Higher lDDT means more similar structures (1.0 is identical).
        
        Args:
            struct1: First structure of shape (seq_len, num_atoms, 3)
            struct2: Second structure of shape (seq_len, num_atoms, 3)
            atom_indices: Optional indices of atoms to consider
            cutoffs: Distance cutoffs for lDDT calculation
            
        Returns:
            lDDT score between the structures
        """
        if struct1.shape != struct2.shape:
            raise ValueError(f"Structures have different shapes: {struct1.shape} vs {struct2.shape}")
        
        # Select atoms if specified
        if atom_indices is not None:
            struct1 = struct1[:, atom_indices, :]
            struct2 = struct2[:, atom_indices, :]
        
        # Reshape to (seq_len * num_atoms, 3)
        struct1_flat = struct1.reshape(-1, 3)
        struct2_flat = struct2.reshape(-1, 3)
        
        # Calculate distance matrices
        n_atoms = struct1_flat.shape[0]
        dm1 = np.zeros((n_atoms, n_atoms))
        dm2 = np.zeros((n_atoms, n_atoms))
        
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                dm1[i, j] = np.sqrt(np.sum((struct1_flat[i] - struct1_flat[j]) ** 2))
                dm2[i, j] = np.sqrt(np.sum((struct2_flat[i] - struct2_flat[j]) ** 2))
                
                # Make the matrices symmetric
                dm1[j, i] = dm1[i, j]
                dm2[j, i] = dm2[i, j]
        
        # Calculate lDDT for each cutoff
        lddt_values = []
        
        for cutoff in cutoffs:
            # Count distances below the cutoff in the reference structure
            ref_distances = dm1[np.triu_indices(n_atoms, k=1)]
            ref_mask = ref_distances < cutoff
            
            if np.sum(ref_mask) == 0:
                # No distances below cutoff
                lddt_values.append(0.0)
                continue
            
            # Count preserved distances
            preserved = 0
            total = 0
            
            for i in range(n_atoms):
                for j in range(i+1, n_atoms):
                    if dm1[i, j] < cutoff:
                        total += 1
                        diff = abs(dm1[i, j] - dm2[i, j])
                        if diff < 0.5 * cutoff:
                            preserved += 1
            
            # Calculate lDDT for this cutoff
            lddt_values.append(preserved / total if total > 0 else 0.0)
        
        # Average lDDT across all cutoffs
        lddt = np.mean(lddt_values)
        
        return lddt
    
    def calculate_tm_score(
        self, 
        struct1: np.ndarray,
        struct2: np.ndarray,
        atom_indices: Optional[List[int]] = None
    ) -> float:
        """
        Calculate the Template Modeling Score (TM-score) between two structures.
        
        Higher TM-score means more similar structures (1.0 is identical).
        
        Args:
            struct1: First structure of shape (seq_len, num_atoms, 3)
            struct2: Second structure of shape (seq_len, num_atoms, 3)
            atom_indices: Optional indices of atoms to consider
            
        Returns:
            TM-score between the structures
        """
        from ..metrics.tm_score import tm_score
        
        if struct1.shape != struct2.shape:
            raise ValueError(f"Structures have different shapes: {struct1.shape} vs {struct2.shape}")
        
        # Select atoms if specified
        if atom_indices is not None:
            struct1 = struct1[:, atom_indices, :]
            struct2 = struct2[:, atom_indices, :]
        
        # Call external TM-score implementation
        # Note: This assumes that tm_score is implemented elsewhere in the codebase
        score = tm_score(struct1, struct2)
        
        return score
    
    def calculate_distance(
        self, 
        struct1: np.ndarray,
        struct2: np.ndarray,
        atom_indices: Optional[List[int]] = None
    ) -> float:
        """
        Calculate the distance between two structures using the specified metric.
        
        Args:
            struct1: First structure of shape (seq_len, num_atoms, 3)
            struct2: Second structure of shape (seq_len, num_atoms, 3)
            atom_indices: Optional indices of atoms to consider
            
        Returns:
            Distance between the structures
        """
        # Use the appropriate distance metric
        if self.distance_metric == "rmsd":
            return self.calculate_rmsd(struct1, struct2, atom_indices)
        elif self.distance_metric == "dme":
            return self.calculate_dme(struct1, struct2, atom_indices)
        elif self.distance_metric == "lddt":
            # Convert lDDT to a distance (lower is more similar)
            lddt = self.calculate_lddt(struct1, struct2, atom_indices)
            return 1.0 - lddt
        elif self.distance_metric == "tm_score":
            # Convert TM-score to a distance (lower is more similar)
            tm = self.calculate_tm_score(struct1, struct2, atom_indices)
            return 1.0 - tm
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
    
    def calculate_distance_matrix(
        self, 
        structures: List[np.ndarray],
        atom_indices: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Calculate the distance matrix between all pairs of structures.
        
        Args:
            structures: List of structures, each of shape (seq_len, num_atoms, 3)
            atom_indices: Optional indices of atoms to consider
            
        Returns:
            Distance matrix of shape (n_structures, n_structures)
        """
        n_structures = len(structures)
        distance_matrix = np.zeros((n_structures, n_structures))
        
        for i in range(n_structures):
            for j in range(i+1, n_structures):
                distance = self.calculate_distance(structures[i], structures[j], atom_indices)
                
                # Make the matrix symmetric
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
        
        return distance_matrix
    
    def calculate_pairwise_diversity(
        self, 
        structures: List[Union[torch.Tensor, np.ndarray]],
        atom_indices: Optional[List[int]] = None
    ) -> Tuple[float, np.ndarray]:
        """
        Calculate the pairwise diversity of a set of structures.
        
        Args:
            structures: List of structures, each of shape (seq_len, num_atoms, 3)
            atom_indices: Optional indices of atoms to consider
            
        Returns:
            Tuple of (average_diversity, distance_matrix)
        """
        # Convert structures to numpy arrays
        numpy_structures = self._convert_structures_to_numpy(structures)
        
        # Calculate distance matrix
        distance_matrix = self.calculate_distance_matrix(numpy_structures, atom_indices)
        
        # Calculate average pairwise diversity
        n_structures = len(structures)
        if n_structures <= 1:
            return 0.0, distance_matrix
        
        # Get the upper triangle of the distance matrix (excluding diagonal)
        upper_triangle = distance_matrix[np.triu_indices(n_structures, k=1)]
        
        # Calculate average
        average_diversity = np.mean(upper_triangle)
        
        return average_diversity, distance_matrix
    
    def _cluster_kmeans(
        self, 
        distance_matrix: np.ndarray,
        n_clusters: int
    ) -> np.ndarray:
        """
        Cluster structures using K-means clustering.
        
        Args:
            distance_matrix: Distance matrix between structures
            n_clusters: Number of clusters to create
            
        Returns:
            Array of cluster labels
        """
        # Use the distance matrix as features for K-means
        # This is a simple approximation, as K-means uses Euclidean distance
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(distance_matrix)
        
        return labels
    
    def _cluster_dbscan(
        self, 
        distance_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Cluster structures using DBSCAN clustering.
        
        Args:
            distance_matrix: Distance matrix between structures
            
        Returns:
            Array of cluster labels
        """
        # Use DBSCAN with precomputed distances
        dbscan = DBSCAN(
            eps=self.cluster_threshold,
            min_samples=2,
            metric="precomputed"
        )
        labels = dbscan.fit_predict(distance_matrix)
        
        return labels
    
    def _cluster_hierarchical(
        self, 
        distance_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Cluster structures using hierarchical clustering.
        
        Args:
            distance_matrix: Distance matrix between structures
            
        Returns:
            Array of cluster labels
        """
        # Convert distance matrix to condensed form
        condensed_distance = squareform(distance_matrix)
        
        # Perform hierarchical clustering
        linkage_matrix = linkage(condensed_distance, method="average")
        
        # Cut the dendrogram at the specified threshold
        labels = fcluster(linkage_matrix, self.cluster_threshold, criterion="distance")
        
        # Convert to zero-based indexing
        labels = labels - 1
        
        return labels
    
    def cluster_structures(
        self, 
        structures: List[Union[torch.Tensor, np.ndarray]],
        n_clusters: Optional[int] = None,
        atom_indices: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cluster structures based on their similarity.
        
        Args:
            structures: List of structures, each of shape (seq_len, num_atoms, 3)
            n_clusters: Number of clusters (required for K-means)
            atom_indices: Optional indices of atoms to consider
            
        Returns:
            Tuple of (cluster_labels, distance_matrix)
        """
        # Convert structures to numpy arrays
        numpy_structures = self._convert_structures_to_numpy(structures)
        
        # Calculate distance matrix
        distance_matrix = self.calculate_distance_matrix(numpy_structures, atom_indices)
        
        # Cluster using the appropriate method
        if self.clustering_method == "kmeans":
            if n_clusters is None:
                n_clusters = min(5, len(structures))
            
            labels = self._cluster_kmeans(distance_matrix, n_clusters)
        elif self.clustering_method == "dbscan":
            labels = self._cluster_dbscan(distance_matrix)
        elif self.clustering_method == "hierarchical":
            labels = self._cluster_hierarchical(distance_matrix)
        else:
            raise ValueError(f"Unknown clustering method: {self.clustering_method}")
        
        return labels, distance_matrix
    
    def select_representatives(
        self, 
        structures: List[Union[torch.Tensor, np.ndarray]],
        n_representatives: int = 5,
        atom_indices: Optional[List[int]] = None
    ) -> List[int]:
        """
        Select representative structures from a set of structures.
        
        Args:
            structures: List of structures, each of shape (seq_len, num_atoms, 3)
            n_representatives: Number of representatives to select
            atom_indices: Optional indices of atoms to consider
            
        Returns:
            List of indices of selected representative structures
        """
        n_structures = len(structures)
        
        # If we have fewer structures than requested representatives, return all
        if n_structures <= n_representatives:
            return list(range(n_structures))
        
        # Convert structures to numpy arrays
        numpy_structures = self._convert_structures_to_numpy(structures)
        
        # Calculate distance matrix
        distance_matrix = self.calculate_distance_matrix(numpy_structures, atom_indices)
        
        # If using K-means, cluster into the desired number of representatives
        if self.clustering_method == "kmeans":
            labels = self._cluster_kmeans(distance_matrix, n_representatives)
        elif self.clustering_method == "dbscan":
            # DBSCAN may produce any number of clusters
            labels = self._cluster_dbscan(distance_matrix)
            
            # If DBSCAN produces too many clusters, use hierarchical
            if len(np.unique(labels)) > n_representatives:
                self.clustering_method = "hierarchical"
                labels = self._cluster_hierarchical(distance_matrix)
        elif self.clustering_method == "hierarchical":
            # Adjust the threshold to get approximately the right number of clusters
            orig_threshold = self.cluster_threshold
            labels = self._cluster_hierarchical(distance_matrix)
            
            # If we get too many clusters, increase the threshold
            while len(np.unique(labels)) > n_representatives:
                self.cluster_threshold *= 1.5
                labels = self._cluster_hierarchical(distance_matrix)
            
            # Restore the original threshold
            self.cluster_threshold = orig_threshold
        else:
            raise ValueError(f"Unknown clustering method: {self.clustering_method}")
        
        # Get unique cluster labels
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        # Select representatives from each cluster
        representatives = []
        
        for label in unique_labels:
            # Get indices of structures in this cluster
            cluster_indices = np.where(labels == label)[0]
            
            if len(cluster_indices) == 0:
                continue
            
            if len(cluster_indices) == 1:
                # If cluster has only one structure, select it
                representatives.append(cluster_indices[0])
            else:
                # Select the structure closest to the cluster center
                cluster_distances = distance_matrix[cluster_indices][:, cluster_indices]
                
                # Calculate average distance to all other structures in the cluster
                avg_distances = np.mean(cluster_distances, axis=1)
                
                # Select the structure with the minimum average distance
                min_idx = np.argmin(avg_distances)
                representatives.append(cluster_indices[min_idx])
        
        # If we don't have enough representatives, add more
        if len(representatives) < n_representatives:
            # Get indices of structures not yet selected
            remaining = list(set(range(n_structures)) - set(representatives))
            
            # Calculate average distance from each remaining structure to all selected ones
            avg_distances = []
            for idx in remaining:
                dist = np.mean([distance_matrix[idx, rep] for rep in representatives])
                avg_distances.append(dist)
            
            # Sort remaining structures by distance (most distant first)
            remaining_sorted = [idx for _, idx in sorted(zip(avg_distances, remaining), reverse=True)]
            
            # Add most distant structures until we have enough representatives
            representatives.extend(remaining_sorted[:n_representatives - len(representatives)])
        
        # If we have too many representatives, select the most diverse subset
        elif len(representatives) > n_representatives:
            # Calculate pairwise distances between representatives
            rep_distances = distance_matrix[representatives][:, representatives]
            
            # Greedy algorithm to select the most diverse subset
            selected = [representatives[0]]  # Start with the first representative
            
            while len(selected) < n_representatives:
                # Calculate minimum distance from each remaining representative to all selected ones
                remaining_reps = list(set(representatives) - set(selected))
                
                min_distances = []
                for idx in remaining_reps:
                    min_dist = min([distance_matrix[idx, sel] for sel in selected])
                    min_distances.append(min_dist)
                
                # Select the representative with the maximum minimum distance
                max_idx = remaining_reps[np.argmax(min_distances)]
                selected.append(max_idx)
            
            representatives = selected
        
        return representatives
    
    def calculate_diversity_metrics(
        self, 
        structures: List[Union[torch.Tensor, np.ndarray]],
        atom_indices: Optional[List[int]] = None
    ) -> Dict[str, float]:
        """
        Calculate various diversity metrics for a set of structures.
        
        Args:
            structures: List of structures, each of shape (seq_len, num_atoms, 3)
            atom_indices: Optional indices of atoms to consider
            
        Returns:
            Dictionary of diversity metrics
        """
        # Convert structures to numpy arrays
        numpy_structures = self._convert_structures_to_numpy(structures)
        
        # Calculate distance matrix
        distance_matrix = self.calculate_distance_matrix(numpy_structures, atom_indices)
        
        # Calculate metrics
        metrics = {}
        
        # Average pairwise distance
        upper_triangle = distance_matrix[np.triu_indices(len(structures), k=1)]
        metrics["avg_pairwise_distance"] = np.mean(upper_triangle) if len(upper_triangle) > 0 else 0.0
        
        # Maximum pairwise distance
        metrics["max_pairwise_distance"] = np.max(upper_triangle) if len(upper_triangle) > 0 else 0.0
        
        # Minimum pairwise distance
        metrics["min_pairwise_distance"] = np.min(upper_triangle) if len(upper_triangle) > 0 else 0.0
        
        # Standard deviation of pairwise distances
        metrics["std_pairwise_distance"] = np.std(upper_triangle) if len(upper_triangle) > 0 else 0.0
        
        # Compute a diversity index (higher means more diverse)
        if len(structures) <= 1:
            metrics["diversity_index"] = 0.0
        else:
            # Normalize by the number of structures
            metrics["diversity_index"] = metrics["avg_pairwise_distance"] * (1.0 - 1.0 / len(structures))
        
        return metrics 