"""
Visualization tools for RNA 3D structures.

This module provides functions to visualize RNA structures in 3D, 
including predicted structures and reference structures for comparison.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d import Axes3D
import py3Dmol
from typing import Dict, List, Optional, Tuple, Union

import torch


def plot_rna_structure_mpl(coords: np.ndarray, 
                          sequence: Optional[List[str]] = None,
                          title: str = "RNA Structure",
                          save_path: Optional[str] = None,
                          show: bool = True):
    """
    Plot RNA structure using matplotlib.
    
    Args:
        coords: Numpy array of shape (seq_len, num_atoms, 3) with 3D coordinates
        sequence: Optional list of nucleotide characters
        title: Title for the plot
        save_path: Path to save the figure (if None, figure is not saved)
        show: Whether to display the figure
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color mapping for nucleotides
    nuc_colors = {
        'A': 'red',
        'U': 'blue',
        'G': 'green',
        'C': 'orange',
        'N': 'gray'
    }
    
    seq_len, num_atoms, _ = coords.shape
    
    # Plot backbone as a line
    backbone_coords = coords[:, 0, :]  # Assuming first atom is backbone phosphate
    ax.plot(backbone_coords[:, 0], backbone_coords[:, 1], backbone_coords[:, 2], 
             '-', color='black', alpha=0.7, linewidth=2, label='Backbone')
    
    # Plot individual nucleotides
    for i in range(seq_len):
        nucleotide_coords = coords[i]
        
        # Get color based on nucleotide type
        if sequence is not None and i < len(sequence):
            color = nuc_colors.get(sequence[i], 'gray')
        else:
            color = 'gray'
        
        # Plot nucleotide atoms
        ax.scatter(nucleotide_coords[:, 0], nucleotide_coords[:, 1], nucleotide_coords[:, 2], 
                   color=color, s=20, alpha=0.8)
        
        # Add label for every 10th nucleotide
        if i % 10 == 0:
            ax.text(nucleotide_coords[0, 0], nucleotide_coords[0, 1], nucleotide_coords[0, 2], 
                    str(i), color='black', fontsize=8)
    
    # Set axis labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Add a legend
    if sequence is not None:
        legend_elements = []
        for nuc, color in nuc_colors.items():
            if nuc in sequence:
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                               markerfacecolor=color, markersize=8, label=nuc))
        ax.legend(handles=legend_elements, loc='upper right')
    
    # Equal aspect ratio
    ax.set_box_aspect((1, 1, 1))
    
    # Save figure if requested
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show or close figure
    if show:
        plt.show()
    else:
        plt.close(fig)


def compare_structures(pred_coords: np.ndarray, 
                       true_coords: np.ndarray,
                       sequence: Optional[List[str]] = None,
                       title: str = "Predicted vs. True Structure",
                       metrics: Optional[Dict[str, float]] = None,
                       save_path: Optional[str] = None,
                       show: bool = True):
    """
    Compare predicted and true RNA structures.
    
    Args:
        pred_coords: Numpy array of predicted coordinates (seq_len, num_atoms, 3)
        true_coords: Numpy array of true coordinates (seq_len, num_atoms, 3)
        sequence: Optional list of nucleotide characters
        title: Title for the plot
        metrics: Optional dictionary of metrics (e.g., RMSD, TM-score)
        save_path: Path to save the figure (if None, figure is not saved)
        show: Whether to display the figure
    """
    fig = plt.figure(figsize=(15, 6))
    
    # Plot predicted structure
    ax1 = fig.add_subplot(121, projection='3d')
    plot_structure_on_axis(ax1, pred_coords, sequence, "Predicted Structure")
    
    # Plot true structure
    ax2 = fig.add_subplot(122, projection='3d')
    plot_structure_on_axis(ax2, true_coords, sequence, "True Structure")
    
    # Add metrics if provided
    if metrics is not None:
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        fig.suptitle(f"{title}\n{metrics_str}")
    else:
        fig.suptitle(title)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if requested
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show or close figure
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_structure_on_axis(ax: plt.Axes, 
                          coords: np.ndarray, 
                          sequence: Optional[List[str]] = None,
                          title: str = ""):
    """
    Helper function to plot RNA structure on a specific matplotlib axis.
    
    Args:
        ax: Matplotlib axis (should be 3D)
        coords: Numpy array of coordinates (seq_len, num_atoms, 3)
        sequence: Optional list of nucleotide characters
        title: Title for the plot
    """
    # Color mapping for nucleotides
    nuc_colors = {
        'A': 'red',
        'U': 'blue',
        'G': 'green',
        'C': 'orange',
        'N': 'gray'
    }
    
    seq_len, num_atoms, _ = coords.shape
    
    # Plot backbone as a line
    backbone_coords = coords[:, 0, :]  # Assuming first atom is backbone phosphate
    ax.plot(backbone_coords[:, 0], backbone_coords[:, 1], backbone_coords[:, 2], 
             '-', color='black', alpha=0.7, linewidth=2)
    
    # Plot individual nucleotides
    for i in range(seq_len):
        nucleotide_coords = coords[i]
        
        # Get color based on nucleotide type
        if sequence is not None and i < len(sequence):
            color = nuc_colors.get(sequence[i], 'gray')
        else:
            color = 'gray'
        
        # Plot nucleotide atoms
        ax.scatter(nucleotide_coords[:, 0], nucleotide_coords[:, 1], nucleotide_coords[:, 2], 
                   color=color, s=20, alpha=0.8)
        
        # Add label for every 10th nucleotide
        if i % 10 == 0:
            ax.text(nucleotide_coords[0, 0], nucleotide_coords[0, 1], nucleotide_coords[0, 2], 
                    str(i), color='black', fontsize=8)
    
    # Set axis labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Equal aspect ratio
    ax.set_box_aspect((1, 1, 1))


def visualize_with_py3Dmol(coords: np.ndarray,
                          sequence: List[str],
                          width: int = 500,
                          height: int = 400,
                          style: str = 'stick'):
    """
    Visualize RNA structure using py3Dmol for interactive visualization.
    
    Args:
        coords: Numpy array of coordinates (seq_len, num_atoms, 3)
        sequence: List of nucleotide characters
        width: Width of the viewer
        height: Height of the viewer
        style: Visualization style ('stick', 'line', 'cartoon', etc.)
        
    Returns:
        py3Dmol viewer object
    """
    # Create PDB format string
    pdb_str = coords_to_pdb(coords, sequence)
    
    # Create viewer
    view = py3Dmol.view(width=width, height=height)
    view.addModel(pdb_str, 'pdb')
    
    # Color by residue
    view.setStyle({style: {'colorscheme': {'A': 'red', 'U': 'blue', 'G': 'green', 'C': 'orange'}}})
    
    # Set camera
    view.zoomTo()
    view.zoom(0.8)
    
    return view


def coords_to_pdb(coords: np.ndarray, sequence: List[str]) -> str:
    """
    Convert coordinates to PDB format string.
    
    Args:
        coords: Numpy array of coordinates (seq_len, num_atoms, 3)
        sequence: List of nucleotide characters
        
    Returns:
        PDB format string
    """
    # Define atom names for each nucleotide
    atom_names = ['P', 'O5\'', 'C5\'', 'C4\'', 'O3\'']
    
    pdb_lines = []
    atom_index = 1
    
    # Add coordinates for each nucleotide
    for i in range(len(sequence)):
        residue_name = sequence[i]
        residue_number = i + 1
        
        # Add atoms for this nucleotide
        for j, atom_name in enumerate(atom_names):
            if j < coords.shape[1]:  # Check if we have coordinates for this atom
                x, y, z = coords[i, j]
                
                # PDB format line
                line = f"ATOM  {atom_index:5d} {atom_name:<4s} {residue_name:3s} A{residue_number:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {atom_name[0]}"
                pdb_lines.append(line)
                atom_index += 1
    
    # Add connecting lines
    for i in range(len(sequence) - 1):
        line = f"CONECT{(i*len(atom_names) + 1):5d}{(i*len(atom_names) + 5):5d}"
        pdb_lines.append(line)
    
    # End of file
    pdb_lines.append("END")
    
    return "\n".join(pdb_lines)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert PyTorch tensor to numpy array.
    
    Args:
        tensor: PyTorch tensor
        
    Returns:
        Numpy array
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    else:
        return tensor 