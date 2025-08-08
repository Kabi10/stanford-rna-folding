#!/usr/bin/env python3
"""
Stanford RNA 3D Folding - Exploratory Data Analysis
===================================================

Comprehensive EDA script for analyzing the Stanford RNA 3D Folding competition dataset.
This script provides data loading, statistical analysis, and visualization capabilities
for RNA sequence and structure data.

Author: Stanford RNA Folding Project
Date: 2025-01-08
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from typing import Dict, List, Tuple, Optional
import json

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class RNAFoldingEDA:
    """
    Exploratory Data Analysis class for Stanford RNA 3D Folding competition data.
    """
    
    def __init__(self, data_path: str = "datasets/stanford-rna-3d-folding"):
        """
        Initialize the EDA class with dataset path.
        
        Args:
            data_path: Path to the competition dataset directory
        """
        self.data_path = Path(data_path)
        self.train_sequences = None
        self.train_labels = None
        self.validation_sequences = None
        self.validation_labels = None
        self.test_sequences = None
        self.sample_submission = None
        
        # RNA nucleotide mapping
        self.nucleotide_map = {'A': 0, 'U': 1, 'G': 2, 'C': 3}
        self.reverse_nucleotide_map = {v: k for k, v in self.nucleotide_map.items()}
        
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all competition datasets.
        
        Returns:
            Dictionary containing all loaded datasets
        """
        print("🔄 Loading Stanford RNA 3D Folding competition data...")
        
        datasets = {}
        
        # Load training data
        train_seq_path = self.data_path / "train_sequences.csv"
        train_labels_path = self.data_path / "train_labels.csv"
        
        if train_seq_path.exists():
            self.train_sequences = pd.read_csv(train_seq_path)
            datasets['train_sequences'] = self.train_sequences
            print(f"✅ Loaded train_sequences: {len(self.train_sequences)} samples")
        
        if train_labels_path.exists():
            self.train_labels = pd.read_csv(train_labels_path)
            datasets['train_labels'] = self.train_labels
            print(f"✅ Loaded train_labels: {len(self.train_labels)} coordinate points")
        
        # Load validation data
        val_seq_path = self.data_path / "validation_sequences.csv"
        val_labels_path = self.data_path / "validation_labels.csv"
        
        if val_seq_path.exists():
            self.validation_sequences = pd.read_csv(val_seq_path)
            datasets['validation_sequences'] = self.validation_sequences
            print(f"✅ Loaded validation_sequences: {len(self.validation_sequences)} samples")
        
        if val_labels_path.exists():
            self.validation_labels = pd.read_csv(val_labels_path)
            datasets['validation_labels'] = self.validation_labels
            print(f"✅ Loaded validation_labels: {len(self.validation_labels)} coordinate points")
        
        # Load test data
        test_seq_path = self.data_path / "test_sequences.csv"
        if test_seq_path.exists():
            self.test_sequences = pd.read_csv(test_seq_path)
            datasets['test_sequences'] = self.test_sequences
            print(f"✅ Loaded test_sequences: {len(self.test_sequences)} samples")
        
        # Load sample submission
        sample_sub_path = self.data_path / "sample_submission.csv"
        if sample_sub_path.exists():
            self.sample_submission = pd.read_csv(sample_sub_path)
            datasets['sample_submission'] = self.sample_submission
            print(f"✅ Loaded sample_submission: {len(self.sample_submission)} entries")
        
        print(f"\n📊 Total datasets loaded: {len(datasets)}")
        return datasets
    
    def analyze_sequences(self) -> Dict:
        """
        Analyze RNA sequence characteristics.
        
        Returns:
            Dictionary containing sequence analysis results
        """
        print("\n🧬 Analyzing RNA Sequences...")
        
        if self.train_sequences is None:
            print("❌ No training sequences loaded!")
            return {}
        
        analysis = {}
        
        # Basic sequence statistics
        sequences = self.train_sequences['sequence'].values
        seq_lengths = [len(seq) for seq in sequences]
        
        analysis['sequence_count'] = len(sequences)
        analysis['length_stats'] = {
            'min': min(seq_lengths),
            'max': max(seq_lengths),
            'mean': np.mean(seq_lengths),
            'median': np.median(seq_lengths),
            'std': np.std(seq_lengths)
        }
        
        # Nucleotide composition analysis
        all_nucleotides = ''.join(sequences)
        nucleotide_counts = {nt: all_nucleotides.count(nt) for nt in 'AUGC'}
        total_nucleotides = sum(nucleotide_counts.values())
        
        analysis['nucleotide_composition'] = {
            nt: count / total_nucleotides for nt, count in nucleotide_counts.items()
        }
        
        # GC content analysis
        gc_contents = []
        for seq in sequences:
            gc_count = seq.count('G') + seq.count('C')
            gc_content = gc_count / len(seq) if len(seq) > 0 else 0
            gc_contents.append(gc_content)
        
        analysis['gc_content_stats'] = {
            'mean': np.mean(gc_contents),
            'std': np.std(gc_contents),
            'min': min(gc_contents),
            'max': max(gc_contents)
        }
        
        print(f"📈 Sequence Analysis Complete:")
        print(f"   • Total sequences: {analysis['sequence_count']}")
        print(f"   • Length range: {analysis['length_stats']['min']}-{analysis['length_stats']['max']}")
        print(f"   • Average length: {analysis['length_stats']['mean']:.1f}")
        print(f"   • GC content: {analysis['gc_content_stats']['mean']:.3f} ± {analysis['gc_content_stats']['std']:.3f}")
        
        return analysis
    
    def analyze_structures(self) -> Dict:
        """
        Analyze 3D structure coordinate data.
        
        Returns:
            Dictionary containing structure analysis results
        """
        print("\n🏗️ Analyzing 3D Structures...")
        
        if self.train_labels is None:
            print("❌ No training labels loaded!")
            return {}
        
        analysis = {}
        
        # Extract unique sequence IDs
        sequence_ids = self.train_labels['ID'].str.split('_').str[0].unique()
        analysis['unique_sequences'] = len(sequence_ids)
        
        # Analyze coordinate statistics
        coords = ['x_1', 'y_1', 'z_1']
        coord_stats = {}

        for coord in coords:
            if coord in self.train_labels.columns:
                values = self.train_labels[coord].values
                coord_stats[coord] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'range': np.max(values) - np.min(values)
                }
        
        analysis['coordinate_stats'] = coord_stats
        
        # Analyze structure sizes (atoms per sequence)
        structure_sizes = []
        for seq_id in sequence_ids:
            seq_coords = self.train_labels[self.train_labels['ID'].str.startswith(seq_id)]
            structure_sizes.append(len(seq_coords))
        
        analysis['structure_size_stats'] = {
            'mean': np.mean(structure_sizes),
            'std': np.std(structure_sizes),
            'min': min(structure_sizes),
            'max': max(structure_sizes)
        }
        
        print(f"📈 Structure Analysis Complete:")
        print(f"   • Unique structures: {analysis['unique_sequences']}")
        print(f"   • Total coordinate points: {len(self.train_labels)}")
        print(f"   • Average atoms per structure: {analysis['structure_size_stats']['mean']:.1f}")
        
        return analysis
    
    def create_visualizations(self, save_path: str = "results/eda_plots") -> None:
        """
        Create comprehensive visualizations of the dataset.
        
        Args:
            save_path: Directory to save plots
        """
        print(f"\n📊 Creating visualizations...")
        
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Sequence length distribution
        if self.train_sequences is not None:
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            seq_lengths = [len(seq) for seq in self.train_sequences['sequence']]
            plt.hist(seq_lengths, bins=30, alpha=0.7, edgecolor='black')
            plt.xlabel('Sequence Length')
            plt.ylabel('Frequency')
            plt.title('Distribution of RNA Sequence Lengths')
            plt.grid(True, alpha=0.3)
            
            # 2. Nucleotide composition
            plt.subplot(1, 2, 2)
            all_nucleotides = ''.join(self.train_sequences['sequence'])
            nucleotide_counts = {nt: all_nucleotides.count(nt) for nt in 'AUGC'}
            
            plt.bar(nucleotide_counts.keys(), nucleotide_counts.values(), 
                   color=['red', 'blue', 'green', 'orange'], alpha=0.7)
            plt.xlabel('Nucleotide')
            plt.ylabel('Count')
            plt.title('Nucleotide Composition')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(save_dir / "sequence_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. 3D coordinate distributions
        if self.train_labels is not None:
            plt.figure(figsize=(15, 5))

            coords = ['x_1', 'y_1', 'z_1']
            colors = ['red', 'green', 'blue']
            coord_labels = ['X', 'Y', 'Z']

            for i, (coord, color, label) in enumerate(zip(coords, colors, coord_labels)):
                if coord in self.train_labels.columns:
                    plt.subplot(1, 3, i+1)
                    plt.hist(self.train_labels[coord], bins=50, alpha=0.7,
                            color=color, edgecolor='black')
                    plt.xlabel(f'{label} Coordinate')
                    plt.ylabel('Frequency')
                    plt.title(f'Distribution of {label} Coordinates')
                    plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(save_dir / "coordinate_distributions.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"✅ Visualizations saved to {save_dir}")
    
    def generate_summary_report(self, output_path: str = "results/eda_summary.json") -> Dict:
        """
        Generate a comprehensive summary report.
        
        Args:
            output_path: Path to save the summary report
            
        Returns:
            Dictionary containing the complete analysis summary
        """
        print("\n📋 Generating Summary Report...")
        
        # Load data if not already loaded
        if self.train_sequences is None:
            self.load_data()
        
        # Perform all analyses
        sequence_analysis = self.analyze_sequences()
        structure_analysis = self.analyze_structures()
        
        # Compile summary report
        summary = {
            'dataset_info': {
                'competition': 'Stanford RNA 3D Folding',
                'analysis_date': pd.Timestamp.now().isoformat(),
                'data_path': str(self.data_path)
            },
            'sequence_analysis': sequence_analysis,
            'structure_analysis': structure_analysis,
            'data_quality': {
                'missing_sequences': self.train_sequences['sequence'].isnull().sum() if self.train_sequences is not None else 0,
                'missing_coordinates': self.train_labels[['x_1', 'y_1', 'z_1']].isnull().sum().to_dict() if self.train_labels is not None else {},
            }
        }
        
        # Save summary report
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"✅ Summary report saved to {output_path}")
        return summary

def main():
    """
    Main function to run the complete EDA analysis.
    """
    print("🚀 Starting Stanford RNA 3D Folding EDA Analysis")
    print("=" * 60)
    
    # Initialize EDA class
    eda = RNAFoldingEDA()
    
    # Load data
    datasets = eda.load_data()
    
    if not datasets:
        print("❌ No datasets found! Please check the data path.")
        return
    
    # Perform analyses
    eda.analyze_sequences()
    eda.analyze_structures()
    
    # Create visualizations
    eda.create_visualizations()
    
    # Generate summary report
    summary = eda.generate_summary_report()
    
    print("\n🎉 EDA Analysis Complete!")
    print("=" * 60)
    print("📁 Check the 'results/' directory for:")
    print("   • eda_plots/ - Visualization plots")
    print("   • eda_summary.json - Detailed analysis report")

if __name__ == "__main__":
    main()
