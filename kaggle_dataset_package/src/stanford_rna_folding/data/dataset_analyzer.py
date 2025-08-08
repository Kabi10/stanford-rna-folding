import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
from tqdm import tqdm
from src.stanford_rna_folding.data.data_processing import StanfordRNADataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RNADatasetAnalyzer:
    """
    Analyzer for RNA 3D dataset to validate data loading and processing.
    """
    
    def __init__(self, dataset: StanfordRNADataset):
        """
        Initialize the analyzer with a dataset.
        
        Args:
            dataset: StanfordRNADataset instance to analyze
        """
        self.dataset = dataset
        self.results = {}
    
    def analyze_dataset(self, sample_size: Optional[int] = None) -> Dict:
        """
        Analyze the entire dataset for issues and statistics.
        
        Args:
            sample_size: Number of samples to analyze (None for all)
            
        Returns:
            Dictionary with analysis results
        """
        logger.info(f"Analyzing {'all' if sample_size is None else sample_size} samples...")
        
        # Get indices to analyze
        if sample_size is None:
            indices = range(len(self.dataset))
        else:
            indices = np.random.choice(len(self.dataset), min(sample_size, len(self.dataset)), replace=False)
        
        # Initialize statistics
        stats = {
            "sequence_lengths": [],
            "coordinate_shapes": [],
            "nan_counts": [],
            "zero_counts": [],
            "mean_values": [],
            "std_values": [],
            "min_values": [],
            "max_values": [],
            "length_mismatches": 0,
            "potential_issues": []
        }
        
        # Analyze each sample
        for idx in tqdm(indices):
            try:
                sample = self.dataset[idx]
                target_id = sample['target_id']
                sequence = sample['sequence_str']
                coordinates = sample['coordinates'].numpy()
                
                # Record basic info
                stats["sequence_lengths"].append(len(sequence))
                stats["coordinate_shapes"].append(coordinates.shape)
                
                # Check for NaNs
                nan_count = np.isnan(coordinates).sum()
                stats["nan_counts"].append(nan_count)
                
                # Check for zeros
                zero_count = np.sum(coordinates == 0.0)
                stats["zero_counts"].append(zero_count)
                
                # Record statistics
                stats["mean_values"].append(np.mean(coordinates))
                stats["std_values"].append(np.std(coordinates))
                stats["min_values"].append(np.min(coordinates))
                stats["max_values"].append(np.max(coordinates))
                
                # Check for length mismatch
                if len(sequence) != coordinates.shape[0]:
                    stats["length_mismatches"] += 1
                    stats["potential_issues"].append({
                        "target_id": target_id,
                        "issue": "length_mismatch",
                        "sequence_length": len(sequence),
                        "coordinate_length": coordinates.shape[0]
                    })
                
                # Check for potential outliers
                mean = np.mean(coordinates)
                std = np.std(coordinates)
                outliers = np.sum(np.abs(coordinates - mean) > 5 * std)
                if outliers > coordinates.size * 0.05:  # More than 5% outliers
                    stats["potential_issues"].append({
                        "target_id": target_id,
                        "issue": "outliers",
                        "outlier_count": outliers,
                        "percentage": outliers/coordinates.size
                    })
                
            except Exception as e:
                logger.error(f"Error analyzing sample {idx}: {str(e)}")
                stats["potential_issues"].append({
                    "index": idx,
                    "issue": "processing_error",
                    "error": str(e)
                })
        
        # Compute summary statistics
        stats["total_samples"] = len(indices)
        stats["avg_sequence_length"] = np.mean(stats["sequence_lengths"])
        stats["min_sequence_length"] = np.min(stats["sequence_lengths"])
        stats["max_sequence_length"] = np.max(stats["sequence_lengths"])
        stats["total_nan_values"] = sum(stats["nan_counts"])
        stats["total_zero_values"] = sum(stats["zero_counts"])
        stats["samples_with_issues"] = len(stats["potential_issues"])
        
        # Store results
        self.results = stats
        logger.info(f"Analysis complete. {stats['samples_with_issues']} samples with potential issues found.")
        
        return stats
    
    def visualize_statistics(self, output_dir: Optional[str] = None) -> None:
        """
        Visualize the dataset statistics.
        
        Args:
            output_dir: Directory to save the plots (None for displaying only)
        """
        if not self.results:
            logger.warning("No analysis results available. Run analyze_dataset() first.")
            return
        
        # Create output directory if needed
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # 1. Sequence length distribution
        plt.figure(figsize=(10, 6))
        plt.hist(self.results["sequence_lengths"], bins=30, alpha=0.7)
        plt.title("RNA Sequence Length Distribution")
        plt.xlabel("Sequence Length")
        plt.ylabel("Count")
        plt.grid(True, alpha=0.3)
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, "sequence_length_dist.png"))
            plt.close()
        else:
            plt.show()
        
        # 2. Zero counts distribution
        plt.figure(figsize=(10, 6))
        zero_percentages = [count/total for count, total in 
                           zip(self.results["zero_counts"], 
                               [np.prod(shape) for shape in self.results["coordinate_shapes"]])]
        plt.hist(zero_percentages, bins=30, alpha=0.7)
        plt.title("Percentage of Zero Values in Coordinates")
        plt.xlabel("Percentage of Zeros")
        plt.ylabel("Count")
        plt.grid(True, alpha=0.3)
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, "zero_values_dist.png"))
            plt.close()
        else:
            plt.show()
        
        # 3. Coordinate value distributions
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        axs[0, 0].hist(self.results["mean_values"], bins=30, alpha=0.7)
        axs[0, 0].set_title("Distribution of Mean Coordinate Values")
        axs[0, 0].set_xlabel("Mean Value")
        axs[0, 0].set_ylabel("Count")
        axs[0, 0].grid(True, alpha=0.3)
        
        axs[0, 1].hist(self.results["std_values"], bins=30, alpha=0.7)
        axs[0, 1].set_title("Distribution of Coordinate Standard Deviations")
        axs[0, 1].set_xlabel("Standard Deviation")
        axs[0, 1].set_ylabel("Count")
        axs[0, 1].grid(True, alpha=0.3)
        
        axs[1, 0].hist(self.results["min_values"], bins=30, alpha=0.7)
        axs[1, 0].set_title("Distribution of Minimum Coordinate Values")
        axs[1, 0].set_xlabel("Minimum Value")
        axs[1, 0].set_ylabel("Count")
        axs[1, 0].grid(True, alpha=0.3)
        
        axs[1, 1].hist(self.results["max_values"], bins=30, alpha=0.7)
        axs[1, 1].set_title("Distribution of Maximum Coordinate Values")
        axs[1, 1].set_xlabel("Maximum Value")
        axs[1, 1].set_ylabel("Count")
        axs[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, "coordinate_statistics.png"))
            plt.close()
        else:
            plt.show()
        
        # 4. Create a summary report
        if output_dir:
            with open(os.path.join(output_dir, "dataset_analysis_report.txt"), "w") as f:
                f.write("RNA Dataset Analysis Report\n")
                f.write("==========================\n\n")
                f.write(f"Total samples analyzed: {self.results['total_samples']}\n")
                f.write(f"Samples with potential issues: {self.results['samples_with_issues']}\n\n")
                
                f.write("Sequence Statistics:\n")
                f.write(f"  Average length: {self.results['avg_sequence_length']:.2f}\n")
                f.write(f"  Min length: {self.results['min_sequence_length']}\n")
                f.write(f"  Max length: {self.results['max_sequence_length']}\n\n")
                
                f.write("Coordinate Statistics:\n")
                f.write(f"  Total NaN values: {self.results['total_nan_values']}\n")
                f.write(f"  Total zero values: {self.results['total_zero_values']}\n")
                f.write(f"  Length mismatches: {self.results['length_mismatches']}\n\n")
                
                f.write("Sample Issues Summary:\n")
                for issue in self.results["potential_issues"][:20]:  # Show first 20 issues
                    f.write(f"  - {issue}\n")
                
                if len(self.results["potential_issues"]) > 20:
                    f.write(f"  - ... and {len(self.results['potential_issues']) - 20} more issues\n")
    
    def visualize_problematic_samples(self, num_samples: int = 5, output_dir: Optional[str] = None) -> None:
        """
        Visualize samples with potential issues for debugging.
        
        Args:
            num_samples: Number of problematic samples to visualize
            output_dir: Directory to save the plots (None for displaying only)
        """
        if not self.results or not self.results.get("potential_issues"):
            logger.warning("No issues found in dataset. Run analyze_dataset() first.")
            return
        
        # Create output directory if needed
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Get problematic samples to visualize
        issues_to_show = min(num_samples, len(self.results["potential_issues"]))
        
        for i in range(issues_to_show):
            issue = self.results["potential_issues"][i]
            
            # Find the sample in the dataset
            target_id = issue.get("target_id")
            if not target_id:
                continue
                
            # Find index by target_id
            idx = None
            for j in range(len(self.dataset)):
                if self.dataset[j]["target_id"] == target_id:
                    idx = j
                    break
            
            if idx is None:
                continue
                
            # Get the sample
            sample = self.dataset[idx]
            sequence = sample['sequence_str']
            coordinates = sample['coordinates'].numpy()
            
            # Create visualization
            fig = plt.figure(figsize=(15, 10))
            fig.suptitle(f"Problematic Sample: {target_id}\nIssue: {issue.get('issue')}", fontsize=16)
            
            # Plot the structure
            ax1 = fig.add_subplot(2, 3, 1, projection='3d')
            ax1.set_title("All Atoms")
            
            # Flatten all atoms for the overview plot
            all_coords = coordinates.reshape(-1, 3)
            ax1.scatter(all_coords[:, 0], all_coords[:, 1], all_coords[:, 2], 
                       c=np.repeat(np.arange(coordinates.shape[0]), coordinates.shape[1]), 
                       cmap='viridis', alpha=0.7)
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Z')
            
            # Plot histogram of coordinate values
            ax2 = fig.add_subplot(2, 3, 2)
            ax2.set_title("Coordinate Value Distribution")
            ax2.hist(all_coords.flatten(), bins=50, alpha=0.7)
            ax2.set_xlabel("Coordinate Value")
            ax2.set_ylabel("Count")
            ax2.grid(True, alpha=0.3)
            
            # Plot sequence information
            ax3 = fig.add_subplot(2, 3, 3)
            ax3.set_title("Sequence Information")
            ax3.axis('off')
            info_text = f"Sequence length: {len(sequence)}\n"
            info_text += f"Coordinate shape: {coordinates.shape}\n"
            info_text += f"Sequence: {sequence[:50]}...\n\n"
            
            if issue.get("issue") == "length_mismatch":
                info_text += f"Length mismatch detected!\n"
                info_text += f"Sequence length: {issue.get('sequence_length')}\n"
                info_text += f"Coordinate length: {issue.get('coordinate_length')}\n"
            elif issue.get("issue") == "outliers":
                info_text += f"Outliers detected!\n"
                info_text += f"Outlier count: {issue.get('outlier_count')}\n"
                info_text += f"Percentage: {issue.get('percentage'):.2%}\n"
            
            ax3.text(0.1, 0.9, info_text, transform=ax3.transAxes, 
                    fontsize=10, verticalalignment='top')
            
            # Plot each atom type separately
            num_atoms = coordinates.shape[1]
            for j in range(min(3, num_atoms)):
                ax = fig.add_subplot(2, 3, j+4, projection='3d')
                ax.set_title(f"Atom {j+1}")
                atom_coords = coordinates[:, j, :]
                ax.scatter(atom_coords[:, 0], atom_coords[:, 1], atom_coords[:, 2], 
                          c=np.arange(coordinates.shape[0]), cmap='viridis')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            
            if output_dir:
                plt.savefig(os.path.join(output_dir, f"issue_{target_id}.png"))
                plt.close()
            else:
                plt.show() 