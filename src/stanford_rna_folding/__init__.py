"""
Stanford RNA 3D Structure Prediction Competition Package

This package contains all the code for the Stanford RNA 3D Structure Prediction
Kaggle competition (https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding).

The goal is to predict the 3D structure (atomic coordinates) of RNA molecules
given their nucleotide sequences.
"""

from . import data

__version__ = "0.1.0"
__all__ = ["data"] 