import numpy as np
import random
import torch # Keep this import, as heatmap typically comes as a torch.Tensor
from numba import jit # <--- ADD THIS LINE

# Original function for distance calculation
# No @jit here usually, as np.linalg.norm is already highly optimized C code.
def calculate_euclidean_distance(coords: np.ndarray) -> np.ndarray:
    """
    Calculates Euclidean distance matrix from coordinates.
    Args:
        coords (np.ndarray): Node coordinates [num_nodes, 2].
    Returns:
        np.ndarray: Distance matrix [num_nodes, num_nodes].
    """
    num_nodes = coords.shape[0]
    distances = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            distances[i, j] = np.linalg.norm(coords[i] - coords[j])
    return distances

# --- Numba JIT-compiled functions for 2-Opt ---

@jit(nopython=True) # <--- ADD THIS DECORATOR
def two_opt_swap(tour: list, i: int, j: int) -> list:
    """
    Performs a 2-opt swap on a tour.
    Reverses the segment of the tour between indices i and j.
    Numba will optimize list slicing and concatenation here.
    """
    new_tour = tour[0:i] + tour[j:i-1:-1] + tour[j+1:]
    return new_tour

@jit(nopython=True) # <--- ADD THIS DECORATOR
def calculate_tour_length(tour: list, dist_matrix: np.ndarray) -> float:
    """Calculates the total length of a given tour."""
    length = 0.0
    # Numba will optimize this loop
    for i in range(len(tour) - 1):
        node1_idx = tour[i]
        node2_idx = tour[i+1]
        length += dist_matrix[node1_idx, node2_idx]
    return length

@jit(nopython=True) # <--- ADD THIS DECORATOR
def two_opt(tour: list, dist_matrix: np.ndarray) -> list:
    """
    Applies the 2-Opt local search algorithm to a tour.

    Args:
        tour (list): The initial tour (sequence of node indices, e.g., [0, 1, 2, 3, 0]).
                    The tour should start and end at the same node.
        dist_matrix (np.ndarray): The precomputed distance matrix between nodes.

    Returns:
        list: The improved tour after applying 2-Opt.
    """
    # Ensure tour is a list of integers for Numba's nopython mode
    best_tour = list(tour)
    best_tour_len = calculate_tour_length(best_tour, dist_matrix)
    num_nodes = len(best_tour) - 1 # Exclude the repeated start/end node

    improved = True
    while improved:
        improved = False
        for i in range(1, num_nodes - 1): # Start from 1 to avoid swapping first edge
            for j in range(i + 1, num_nodes): # End before last node
                if j - i == 1: continue # Skip adjacent edges

                # Perform the swap and calculate new length
                new_tour = two_opt_swap(best_tour, i, j)
                new_tour_len = calculate_tour_length(new_tour, dist_matrix)

                # If an improvement is found, update and restart the search
                if new_tour_len < best_tour_len:
                    best_tour = new_tour
                    best_tour_len = new_tour_len
                    improved = True
                    # Breaks to restart the entire search from the improved tour.
                    # This is a common strategy to ensure global improvement.
                    break
            if improved:
                break
    return best_tour

# Function to extract tour from heatmap and apply 2-Opt
def generate_tour_from_heatmap_and_coords(heatmap: torch.Tensor, coords: np.ndarray) -> tuple[list, float]:
    """
    Extracts a tour from a heatmap using a greedy approach and refines it using 2-Opt.
    """
    num_nodes = heatmap.shape[0]

    # Convert heatmap to NumPy array for processing (if it's a PyTorch tensor)
    if isinstance(heatmap, torch.Tensor):
        heatmap_np = heatmap.detach().cpu().numpy()
    else:
        heatmap_np = heatmap

    # Step 1: Greedy tour extraction
    initial_tour = []
    visited = [False] * num_nodes
    current_node = random.randint(0, num_nodes - 1) # Start from a random node
    initial_tour.append(current_node)
    visited[current_node] = True

    for _ in range(num_nodes - 1):
        # Find the next unvisited node with the highest probability from the heatmap
        next_probabilities = heatmap_np[current_node, :].copy()
        # Set visited nodes to -inf probability to exclude them
        for i in range(num_nodes):
            if visited[i]:
                next_probabilities[i] = -np.inf

        # Handle cases where all remaining nodes might have -inf (shouldn't happen with proper heatmap)
        if np.all(next_probabilities == -np.inf):
            break # No valid next node found

        next_node = np.argmax(next_probabilities)
        initial_tour.append(next_node)
        visited[next_node] = True
        current_node = next_node

    # Ensure the tour is a cycle by returning to the start node
    if len(initial_tour) == num_nodes and initial_tour[0] != initial_tour[-1]:
         initial_tour.append(initial_tour[0]) # Make it a cycle

    # Basic validity check for a complete tour
    if len(set(initial_tour[:-1])) != num_nodes or len(initial_tour) != num_nodes + 1:
        # This indicates a failure in greedy extraction, potentially due to poor heatmap
        # or a very complex graph structure.
        print(f"Warning: Greedy tour extraction failed to form a complete tour for N={num_nodes}. "
              "Returning infinite length.")
        return initial_tour, float('inf') # Indicate a failed tour

    # Step 2: Apply 2-Opt local search
    # Calculate distance matrix (this part is outside Numba's jit)
    distances = calculate_euclidean_distance(coords)

    # Ensure initial_tour is a list of integers before passing to JIT-compiled two_opt
    initial_tour_list = [int(node) for node in initial_tour]
    improved_tour = two_opt(initial_tour_list, distances)

    # Step 3: Calculate final tour length using the JIT-compiled function
    final_length = calculate_tour_length(improved_tour, distances)

    return improved_tour, final_length