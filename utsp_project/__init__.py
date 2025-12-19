"""UTSP Project - Unsupervised GNN Solver for the Travelling Salesman Problem

This package contains the core implementation of the UTSP framework including:
- Graph Neural Network models (GAT-based architecture)
- Loss functions (baseline and alternative with Fiedler value)
- Dataset generation utilities
- Tour construction and optimization algorithms
"""

from .model import UTSPGNN
from .dataset import generate_tsp_instance
from .loss import calculate_utsp_loss
from .tsp_env import generate_tour_from_heatmap_and_coords

__version__ = "1.0.0"
__author__ = "Joffin Koshy"
