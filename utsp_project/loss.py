# loss.py
import torch
import torch.nn.functional as F


def calculate_distance_matrix(coords: torch.Tensor) -> torch.Tensor:
    """
    Calculates the pairwise Euclidean distance matrix for a batch of coordinates.
    Args:
        coords: A torch.Tensor of shape [batch_size, num_nodes, 2].
    Returns:
        A torch.Tensor of shape [batch_size, num_nodes, num_nodes] with distances.
    """
    # torch.cdist efficiently computes the Euclidean distance between all pairs of points
    # in the input tensors.
    return torch.cdist(coords, coords)


def calculate_utsp_loss(heatmaps: torch.Tensor, coords: torch.Tensor, C1_penalty: float) -> torch.Tensor:
    """
    Calculates the total unsupervised loss as described in the UTSP paper (Section 4.4).
    L_total = L_length + C1_penalty * L_cycle

    Args:
        heatmaps: Predicted edge probabilities from GNN, shape [batch_size, num_nodes, num_nodes].
                  Values should be between 0 and 1 (e.g., after sigmoid).
        coords: Original city coordinates, shape [batch_size, num_nodes, 2].
        C1_penalty: Hyperparameter (scalar float) for the L_cycle component, weighting its importance.
                    (e.g., 20.0 for TSP200 from README.md)
    Returns:
        A scalar torch.Tensor representing the total unsupervised loss for the batch.
    """
    batch_size, num_nodes, _ = coords.shape

    # --- 1. Calculate L_length (Expected Tour Length) ---
    # This component minimizes the total length of the tour.
    # It's the sum of (probability of an edge) * (length of that edge) for all possible edges.

    # Get the pairwise Euclidean distance matrix for the current batch of TSP instances
    # Shape: [batch_size, num_nodes, num_nodes]
    dist_matrix = calculate_distance_matrix(coords)

    # Element-wise multiplication of heatmap (probabilities) and distance matrix.
    # This gives the expected length contribution of each edge.
    # Shape: [batch_size, num_nodes, num_nodes]
    expected_edge_lengths = heatmaps * dist_matrix

    # Sum up all expected edge lengths for each TSP instance in the batch.
    # .sum(dim=(-1, -2)) sums across the last two dimensions (num_nodes x num_nodes).
    # This results in a tensor of shape [batch_size].
    total_expected_lengths_per_instance = expected_edge_lengths.sum(dim=(-1, -2))

    # Take the mean across the batch to get a single scalar value for L_length.
    L_length = total_expected_lengths_per_instance.mean()

    # --- 2. Calculate L_cycle (Hamiltonian Cycle Constraint) ---
    # This component enforces that the predicted heatmap forms a valid Hamiltonian cycle.
    # A valid Hamiltonian cycle implies that each node must have exactly one incoming and
    # exactly one outgoing edge. Thus, the sum of probabilities for each row and each column
    # in the heatmap should ideally be 1.
    # This is based on Equation (4) in Section 4.4 of the paper.

    # Calculate row sums: sum of outgoing edge probabilities from each node.
    # Sum along the last dimension (columns) -> [batch_size, num_nodes]
    row_sums = heatmaps.sum(dim=-1)

    # Calculate column sums: sum of incoming edge probabilities to each node.
    # Sum along the second-to-last dimension (rows) -> [batch_size, num_nodes]
    col_sums = heatmaps.sum(dim=-2)

    # Create a target tensor of ones with the same shape as row_sums (and col_sums).
    # This is what the row_sums and col_sums should ideally be.
    ones_target = torch.ones_like(row_sums)

    # Calculate Mean Squared Error (MSE) for deviations from the target of 1.
    # L_cycle_row_penalty penalizes if row sums are not 1.
    L_cycle_row_penalty = F.mse_loss(row_sums, ones_target)

    # L_cycle_col_penalty penalizes if column sums are not 1.
    L_cycle_col_penalty = F.mse_loss(col_sums, ones_target)

    # The total L_cycle is the sum of row and column penalties.
    L_cycle = L_cycle_row_penalty + L_cycle_col_penalty

    # --- 3. Calculate Total Loss ---
    # The total loss is the sum of L_length and the weighted L_cycle.
    L_total = L_length + C1_penalty * L_cycle

    return L_total


if __name__ == "__main__":
    # --- Test Block for loss.py ---
    # This block runs only when loss.py is executed directly.
    # It demonstrates how to use the functions and verifies they produce a scalar output.

    # Determine the device (CPU or GPU) for computation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing loss.py on device: {device}")

    # Define test parameters
    batch_size_test = 4  # Number of TSP instances in the batch
    num_nodes_test = 20  # Number of cities in each TSP instance
    C1_penalty_test = 20.0  # Example C1_penalty hyperparameter (from README.md for TSP 200)

    # 1. Create dummy heatmaps (simulating output from the GNN model)
    # We'll use random values between 0 and 1, as sigmoid output.
    # Using .uniform_ to get a wider range of values for a more robust test.
    dummy_heatmaps = torch.empty(batch_size_test, num_nodes_test, num_nodes_test, device=device).uniform_(0.0, 1.0)

    # Optional: For a strict TSP, a node cannot connect to itself (no self-loops).
    # You might want to explicitly set the diagonal of the heatmap to 0.
    # This might make the L_cycle more representative if your model won't predict self-loops.
    # For i in range(num_nodes_test):
    #     dummy_heatmaps[:, i, i] = 0.0 # Set diagonal probabilities to 0

    # 2. Create dummy coordinates (simulating input to the GNN model)
    # Random 2D coordinates between 0 and 1.
    dummy_coords = torch.rand(batch_size_test, num_nodes_test, 2, device=device)

    # Calculate the total loss using the defined function
    total_loss = calculate_utsp_loss(dummy_heatmaps, dummy_coords, C1_penalty_test)

    # Print the test output
    print(f"\n--- Loss Test Output ---")
    print(f"Dummy Heatmap shape: {dummy_heatmaps.shape}")
    print(f"Dummy Coords shape: {dummy_coords.shape}")
    print(f"Calculated Total Loss: {total_loss.item():.4f}")  # .item() converts scalar tensor to Python number