"""
Alignment module for RNA structure evaluation.

This module provides utilities for structural alignment, including:
1. Integration with US-align for sequence-independent structure alignment
2. Functions for processing alignment results
3. Helper utilities for preparing structures for alignment
"""

import os
import subprocess
import tempfile
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from .metrics import tm_score


class USAlignError(Exception):
    """Exception raised when US-align fails."""
    pass


def save_pdb_from_coords(
    coords: Union[np.ndarray, Tensor],
    output_path: str,
    atom_names: Optional[List[str]] = None,
    residue_names: Optional[List[str]] = None
) -> None:
    """
    Save coordinates to a PDB file for use with external tools like US-align.
    
    Args:
        coords: Coordinates tensor of shape (N, 3) or (N, A, 3)
        output_path: Path to save the PDB file
        atom_names: Optional list of atom names (default: ["P", "O5'", "C5'", "C4'", "C3'"])
        residue_names: Optional list of residue names (default: repeating "A", "C", "G", "U")
    """
    # Convert torch tensor to numpy if needed
    if isinstance(coords, Tensor):
        coords = coords.detach().cpu().numpy()
    
    # Default atom names for RNA (5 atoms per nucleotide)
    if atom_names is None:
        atom_names = ["P", "O5'", "C5'", "C4'", "C3'"]
    
    # Default residue names (repeating nucleotides)
    if residue_names is None:
        residue_types = ["A", "C", "G", "U"]
        if len(coords.shape) == 3:  # (N, A, 3) format
            num_residues = coords.shape[0]
        else:  # (N, 3) format, assuming one atom per residue
            num_residues = coords.shape[0]
            
        residue_names = [residue_types[i % len(residue_types)] for i in range(num_residues)]
    
    # Open file for writing
    with open(output_path, 'w') as f:
        atom_index = 1
        
        # Handle different coordinate shapes
        if len(coords.shape) == 3:  # (N, A, 3) format
            num_residues, num_atoms, _ = coords.shape
            for i in range(num_residues):
                residue_name = residue_names[i]
                for j in range(num_atoms):
                    atom_name = atom_names[j] if j < len(atom_names) else f"A{j}"
                    x, y, z = coords[i, j]
                    # PDB format
                    f.write(f"ATOM  {atom_index:5d} {atom_name:<4s} {residue_name:3s} A{i+1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n")
                    atom_index += 1
        else:  # (N, 3) format
            num_points = coords.shape[0]
            for i in range(num_points):
                # Determine residue index (for PDB, multiple atoms can belong to same residue)
                residue_index = i // len(atom_names) if atom_names else i
                residue_name = residue_names[residue_index] if residue_index < len(residue_names) else "UNK"
                
                # Determine atom name
                atom_index_in_residue = i % len(atom_names) if atom_names else 0
                atom_name = atom_names[atom_index_in_residue] if atom_names and atom_index_in_residue < len(atom_names) else "CA"
                
                x, y, z = coords[i]
                # PDB format
                f.write(f"ATOM  {atom_index:5d} {atom_name:<4s} {residue_name:3s} A{residue_index+1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n")
                atom_index += 1
        
        f.write("END\n")


def run_us_align(
    coords1: Union[np.ndarray, Tensor],
    coords2: Union[np.ndarray, Tensor],
    us_align_path: Optional[str] = None,
    timeout: int = 60,
    **kwargs
) -> Dict:
    """
    Run US-align on two structures to get sequence-independent alignment and TM-score.
    
    Args:
        coords1: First structure coordinates, shape (N, 3) or (N, A, 3)
        coords2: Second structure coordinates, shape (M, 3) or (M, A, 3)
        us_align_path: Path to US-align executable (if None, assumes it's in PATH)
        timeout: Timeout in seconds for US-align process
        **kwargs: Additional arguments to pass to US-align
        
    Returns:
        Dictionary containing alignment results including:
            - tm_score: The TM-score
            - aligned_coords1: Coordinates of structure 1 after alignment
            - aligned_coords2: Coordinates of structure 2 after alignment
            - rotation_matrix: Rotation matrix used for alignment
            - translation_vector: Translation vector used for alignment
    """
    # Convert torch tensors to numpy if needed
    if isinstance(coords1, Tensor):
        coords1 = coords1.detach().cpu().numpy()
    if isinstance(coords2, Tensor):
        coords2 = coords2.detach().cpu().numpy()
    
    # Create temporary directory for input/output files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save structures to PDB files
        pdb1_path = os.path.join(temp_dir, 'structure1.pdb')
        pdb2_path = os.path.join(temp_dir, 'structure2.pdb')
        
        save_pdb_from_coords(coords1, pdb1_path)
        save_pdb_from_coords(coords2, pdb2_path)
        
        # Output files
        matrix_path = os.path.join(temp_dir, 'alignment_matrix.txt')
        aligned_pdb1_path = os.path.join(temp_dir, 'aligned1.pdb')
        aligned_pdb2_path = os.path.join(temp_dir, 'aligned2.pdb')
        
        # Construct US-align command
        cmd = [us_align_path if us_align_path else 'USalign']
        cmd.extend([pdb1_path, pdb2_path])
        cmd.extend(['-mol', 'RNA'])  # Specify RNA molecule type
        cmd.extend(['-outfmt', '2'])  # Detailed output format
        cmd.extend(['-matrix', matrix_path])  # Save alignment matrix
        cmd.extend(['-o', aligned_pdb1_path])  # Output aligned structure 1
        cmd.extend(['-atom', 'P'])  # Use P atoms for RNA alignment
        
        # Add any additional arguments
        for key, value in kwargs.items():
            cmd.extend([f'-{key}', str(value)])
        
        # Run US-align
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=True
            )
        except subprocess.TimeoutExpired:
            raise USAlignError(f"US-align process timed out after {timeout} seconds")
        except subprocess.CalledProcessError as e:
            raise USAlignError(f"US-align failed with exit code {e.returncode}. Error: {e.stderr}")
        
        # Parse US-align output
        return parse_us_align_output(
            result.stdout,
            matrix_path,
            aligned_pdb1_path,
            aligned_pdb2_path
        )


def parse_us_align_output(
    stdout: str,
    matrix_path: str,
    aligned_pdb1_path: str,
    aligned_pdb2_path: str
) -> Dict:
    """
    Parse the output from US-align.
    
    Args:
        stdout: Standard output from US-align
        matrix_path: Path to the alignment matrix file
        aligned_pdb1_path: Path to the aligned structure 1
        aligned_pdb2_path: Path to the aligned structure 2
        
    Returns:
        Dictionary containing parsed results
    """
    # Initialize results dictionary
    results = {}
    
    # Extract TM-score
    for line in stdout.split('\n'):
        if 'TM-score' in line:
            # Format typically: "TM-score = 0.7890 (normalized by length of structure1)"
            parts = line.split('=')
            if len(parts) >= 2:
                tm_score_value = float(parts[1].split('(')[0].strip())
                results['tm_score'] = tm_score_value
                break
    
    # Read alignment matrix (rotation + translation)
    with open(matrix_path, 'r') as f:
        matrix_lines = f.readlines()
    
    rotation_matrix = np.zeros((3, 3))
    translation_vector = np.zeros(3)
    
    # Parse rotation matrix and translation vector
    for i in range(3):
        if i < len(matrix_lines):
            parts = matrix_lines[i].strip().split()
            if len(parts) >= 3:
                rotation_matrix[i, 0] = float(parts[0])
                rotation_matrix[i, 1] = float(parts[1])
                rotation_matrix[i, 2] = float(parts[2])
                translation_vector[i] = float(parts[3])
    
    results['rotation_matrix'] = rotation_matrix
    results['translation_vector'] = translation_vector
    
    # Read aligned structures
    # This would be a more complex parser for PDB files
    # For simplicity, we're just checking if the files exist
    results['aligned_pdb1_path'] = aligned_pdb1_path if os.path.exists(aligned_pdb1_path) else None
    results['aligned_pdb2_path'] = aligned_pdb2_path if os.path.exists(aligned_pdb2_path) else None
    
    return results


def tm_score_with_us_align(
    pred_coords: Union[np.ndarray, Tensor],
    true_coords: Union[np.ndarray, Tensor],
    us_align_path: Optional[str] = None,
    d0: Optional[float] = None,
    fallback_to_internal: bool = True
) -> float:
    """
    Calculate TM-score using US-align for sequence-independent alignment.
    Falls back to internal implementation if US-align fails or is not available.
    
    Args:
        pred_coords: Predicted coordinates
        true_coords: True coordinates
        us_align_path: Path to US-align executable
        d0: Distance scaling factor (if None, calculated according to competition rules)
        fallback_to_internal: Whether to fall back to internal TM-score if US-align fails
        
    Returns:
        TM-score value
    """
    try:
        # Try using US-align
        result = run_us_align(pred_coords, true_coords, us_align_path=us_align_path)
        return result['tm_score']
    except (USAlignError, FileNotFoundError) as e:
        if fallback_to_internal:
            print(f"Warning: US-align failed ({str(e)}). Falling back to internal TM-score implementation.")
            # Fall back to internal implementation
            if isinstance(pred_coords, np.ndarray):
                pred_coords = torch.from_numpy(pred_coords).float()
            if isinstance(true_coords, np.ndarray):
                true_coords = torch.from_numpy(true_coords).float()
                
            return tm_score(pred_coords, true_coords, d0=d0, align=True).item()
        else:
            raise 