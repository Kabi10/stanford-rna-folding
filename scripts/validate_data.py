import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.stanford_rna_folding.data.data_processing import StanfordRNADataset
from src.stanford_rna_folding.data.dataset_analyzer import RNADatasetAnalyzer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Validate data loading and processing for RNA 3D folding dataset."""
    parser = argparse.ArgumentParser(description='Validate RNA dataset loading and processing')
    parser.add_argument('--data_dir', type=str, default='datasets/stanford-rna-3d-folding', 
                        help='Directory containing competition data')
    parser.add_argument('--output_dir', type=str, default='analysis/data_validation', 
                        help='Directory to save analysis results')
    parser.add_argument('--sample_size', type=int, default=100, 
                        help='Number of samples to analyze (use -1 for all)')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'validation', 'test'],
                        help='Data split to analyze')
    args = parser.parse_args()
    
    # Convert -1 to None for analyzing all samples
    sample_size = None if args.sample_size == -1 else args.sample_size
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize the dataset
    logger.info(f"Loading {args.split} dataset from {args.data_dir}...")
    
    dataset = StanfordRNADataset(
        data_dir=args.data_dir,
        split=args.split,
        transform=None
    )
    
    logger.info(f"Dataset loaded with {len(dataset)} samples")
    
    # Analyze the dataset
    logger.info("Starting dataset analysis...")
    analyzer = RNADatasetAnalyzer(dataset)
    stats = analyzer.analyze_dataset(sample_size=sample_size)
    
    # Log summary statistics
    logger.info(f"Analysis complete. Summary:")
    logger.info(f"  Total samples analyzed: {stats['total_samples']}")
    logger.info(f"  Samples with potential issues: {stats['samples_with_issues']}")
    logger.info(f"  Average sequence length: {stats['avg_sequence_length']:.2f}")
    logger.info(f"  Total NaN values: {stats['total_nan_values']}")
    logger.info(f"  Total zero values: {stats['total_zero_values']}")
    logger.info(f"  Length mismatches: {stats['length_mismatches']}")
    
    # Generate visualizations and reports
    logger.info(f"Generating visualizations and reports in {args.output_dir}...")
    analyzer.visualize_statistics(output_dir=args.output_dir)
    
    # Visualize problematic samples
    if stats['samples_with_issues'] > 0:
        logger.info(f"Visualizing problematic samples...")
        analyzer.visualize_problematic_samples(
            num_samples=min(10, stats['samples_with_issues']),
            output_dir=os.path.join(args.output_dir, "problematic_samples")
        )
    
    logger.info(f"Validation complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 