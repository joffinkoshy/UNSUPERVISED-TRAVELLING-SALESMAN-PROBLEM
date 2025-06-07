# dataset.py
import torch

def generate_tsp_instance(num_nodes: int, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """
    Generates a random Euclidean TSP instance.
    Args:
        num_nodes: The number of cities in the TSP instance.
        device: The device (CPU or GPU) to place the generated tensor on.
    Returns:
        A torch.Tensor of shape [num_nodes, 2] with random city coordinates.
    """
    # Your implementation here: generate random coordinates
    coords = torch.rand(num_nodes, 2, device=device)
    return coords

if __name__ == "__main__":
    # This block runs only when dataset.py is executed directly
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    num_cities = 20
    tsp_coords = generate_tsp_instance(num_cities, device=device)
    print(f"Generated TSP instance (first 5 cities):\n{tsp_coords[:5]}")
    print(f"Shape of TSP instance: {tsp_coords.shape}")