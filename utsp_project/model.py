import torch
import torch.nn as nn
import torch.nn.functional as F


# Helper for initial coordinate embedding
class GraphEmbedding(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int):
        super(GraphEmbedding, self).__init__()
        self.embedding_layer = nn.Linear(input_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is [batch_size, num_nodes, input_dim (2 for coords)]
        # Output will be [batch_size, num_nodes, embedding_dim]
        return self.embedding_layer(x)


# GATLayer implementation
class GATLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_heads: int, dropout: float, alpha: float = 0.2):
        super(GATLayer, self).__init__()
        self.output_dim = output_dim  # D_out per head
        self.num_heads = num_heads  # H
        self.alpha = alpha  # Negative slope of LeakyReLU

        # Linear transformation for each head for self and neighbors
        # Maps input_dim -> output_dim * num_heads
        self.W = nn.Parameter(torch.empty(size=(input_dim, output_dim * num_heads)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # Gain for ReLU

        # Attention mechanisms for each head. 'a' has shape [num_heads, 2 * output_dim, 1]
        self.a = nn.Parameter(torch.empty(size=(num_heads, 2 * output_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj: torch.Tensor):
        # h: [batch_size, num_nodes, input_dim]
        # adj: [batch_size, num_nodes, num_nodes] (adjacency matrix, 0 or 1)

        batch_size, num_nodes, _ = h.shape

        # Wh: [B, N, H, D_out]
        Wh = torch.matmul(h, self.W).view(batch_size, num_nodes, self.num_heads, self.output_dim)

        # Prepare for attention computation:
        # Expand Wh for all (i, j) pairs. Resulting shape: [B, N_i, N_j, H, D_out]
        # Wh_i_expanded: current node features expanded for all possible neighbors j
        Wh_i_expanded = Wh.unsqueeze(2).expand(batch_size, num_nodes, num_nodes, self.num_heads, self.output_dim)
        # Wh_j_expanded: neighbor node features expanded for all possible current nodes i
        Wh_j_expanded = Wh.unsqueeze(1).expand(batch_size, num_nodes, num_nodes, self.num_heads, self.output_dim)

        # Concatenate features for attention computation: [B, N, N, H, 2 * D_out]
        concat_features = torch.cat([Wh_i_expanded, Wh_j_expanded], dim=-1)

        # Calculate attention scores using einsum
        # 'bijhd' for concat_features (B, N_i, N_j, H, 2*D_out)
        # 'hdk' for self.a (H, 2*D_out, 1)
        # Result 'bijhk' (B, N_i, N_j, H, 1) -> squeeze(-1) to (B, N_i, N_j, H)
        # The equation for GAT attention coefficients a_{ij} = LeakyReLU(a^T [W h_i || W h_j])
        # This translates to: for each head 'h', and for each pair (i,j), multiply the concatenated features by the attention vector 'a' for that head.
        attention_scores = self.leakyrelu(torch.einsum('bijhd,hdk->bijhk', concat_features, self.a)).squeeze(
            -1)  # [B, N, N, H]

        # Permute attention_scores to [B, H, N, N] for masked_fill and softmax over neighbors
        attention_scores_perm = attention_scores.permute(0, 3, 1, 2).contiguous()  # [B, H, N_i, N_j]

        # Apply adjacency mask: adj is [B, N, N]. Need to expand to [B, 1, N, N]
        neg_inf = -9e15  # A large negative number
        # Set attention scores to very small negative number for non-neighbors (masked out)
        attention_scores_masked = attention_scores_perm.masked_fill(adj.unsqueeze(1) == 0, neg_inf)

        # Apply softmax to get attention weights: [B, H, N, N] (softmax over j)
        attention_weights = F.softmax(attention_scores_masked, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Aggregate neighbor features based on attention weights
        # Wh (used for aggregation) is [B, N, H, D_out]. Permute to [B, H, N, D_out] for matmul
        Wh_agg = Wh.permute(0, 2, 1, 3).contiguous()  # [B, H, N_j, D_out] for aggregation (permute N and H)

        # h_prime = [B, H, N_i, D_out] (attention_weights [B, H, N_i, N_j] @ Wh_agg [B, H, N_j, D_out])
        h_prime = torch.matmul(attention_weights, Wh_agg)

        # Concatenate outputs from different attention heads
        # h_prime: [B, H, N_i, D_out] -> [B, N_i, H*D_out]
        h_prime = h_prime.permute(0, 2, 1, 3).contiguous().view(batch_size, num_nodes, self.num_heads * self.output_dim)

        return h_prime


# UTSPGNN model using the GATLayer
class UTSPGNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, num_heads: int, dropout: float = 0.0):
        super(UTSPGNN, self).__init__()
        self.num_layers = num_layers
        self.embedding_layer = GraphEmbedding(input_dim, hidden_dim)

        self.gnn_layers = nn.ModuleList()
        # Input to first GATLayer is embedding_dim (hidden_dim), output is hidden_dim (num_heads * output_dim)
        # output_dim for a single head is hidden_dim // num_heads
        self.gnn_layers.append(GATLayer(hidden_dim, hidden_dim // num_heads, num_heads, dropout))
        for _ in range(num_layers - 1):
            # Subsequent layers take the concatenated output of previous layer (hidden_dim) as input
            self.gnn_layers.append(GATLayer(hidden_dim, hidden_dim // num_heads, num_heads, dropout))

            # Output layer to transform final GNN representation to heatmap logits
        # Corrected: input features to output_layer should be 2 * hidden_dim
        self.output_layer = nn.Linear(2 * hidden_dim, 1)  # Output for each pair is a single logit

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: [batch_size, num_nodes, 2]
        batch_size, num_nodes, _ = coords.shape

        # Step 1: Embed coordinates
        h = self.embedding_layer(coords)  # h: [batch_size, num_nodes, hidden_dim]

        # Create a full adjacency matrix for the GNN (fully connected graph)
        adj = torch.ones(batch_size, num_nodes, num_nodes, device=coords.device)
        for i in range(num_nodes):
            adj[:, i, i] = 0

        # Step 2: Pass through GNN layers
        for i, gnn_layer in enumerate(self.gnn_layers):
            h = gnn_layer(h, adj)
            if i < self.num_layers - 1:  # Apply ReLU and dropout after intermediate layers
                h = F.relu(h)
                h = F.dropout(h, training=self.training)

        # Step 3: Compute pairwise edge scores for heatmap
        # h after GNN layers: [B, N, hidden_dim]
        h_i = h.unsqueeze(2).repeat(1, 1, num_nodes, 1)  # [B, N, N, hidden_dim]
        h_j = h.unsqueeze(1).repeat(1, num_nodes, 1, 1)  # [B, N, N, hidden_dim]

        edge_features = torch.cat((h_i, h_j), dim=-1)  # [B, N, N, 2 * hidden_dim]

        logits = self.output_layer(edge_features).squeeze(-1)  # [B, N, N]
        heatmap = torch.sigmoid(logits)

        # --- FIX: Ensure diagonal elements are 0 without inplace modification ---
        # Create a mask with zeros on the diagonal and ones elsewhere
        # Multiply heatmap by this mask to set diagonals to zero without inplace modification
        diagonal_mask = (1 - torch.eye(num_nodes, device=heatmap.device)).unsqueeze(0)
        heatmap = heatmap * diagonal_mask
        # The previous problematic loop:
        # for i in range(num_nodes):
        #     heatmap[:, i, i] = 0.0

        return heatmap


# Test block for model.py (will run if model.py is executed directly)
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing model.py with GATLayer on device: {device}")

    batch_size_test = 2
    num_nodes_test = 5  # Small number for testing
    input_dim_test = 2  # x, y coordinates
    hidden_dim_test = 64
    num_layers_test = 2
    num_heads_test = 8
    dropout_test = 0.1

    model = UTSPGNN(input_dim_test, hidden_dim_test, num_layers_test, num_heads_test, dropout_test).to(device)
    model.eval()  # Set to eval mode for consistent dropout behavior during test

    # Dummy coordinates for a batch of TSP instances
    dummy_coords = torch.rand(batch_size_test, num_nodes_test, input_dim_test, device=device)

    print(f"\n--- Model Test Output (GATLayer) ---")
    print(f"Dummy Coords shape: {dummy_coords.shape}")

    with torch.no_grad():
        heatmap_output = model(dummy_coords)

    print(f"Heatmap Output shape: {heatmap_output.shape}")
    print(f"Heatmap values (first instance, top-left 3x3):\n{heatmap_output[0, :3, :3]}")

    # Check heatmap properties
    assert heatmap_output.shape == (batch_size_test, num_nodes_test, num_nodes_test)
    assert (heatmap_output >= 0.0).all() and (heatmap_output <= 1.0).all()
    # Check diagonal elements are zero
    for i in range(num_nodes_test):
        assert torch.isclose(heatmap_output[:, i, i], torch.tensor(0.0).to(device)).all()

    print("\nModel with GATLayer tests completed successfully!")