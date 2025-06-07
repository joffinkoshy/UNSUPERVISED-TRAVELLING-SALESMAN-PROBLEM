# train.py
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from model import UTSPGNN
from dataset import generate_tsp_instance

try:
    from loss import calculate_utsp_loss as baseline_calculate_utsp_loss
    BASELINE_LOSS_AVAILABLE = True
except ImportError:
    BASELINE_LOSS_AVAILABLE = False
    print("Warning: Could not import 'calculate_utsp_loss' from loss.py. 'baseline' loss type will not be available.")
    def baseline_calculate_utsp_loss(*args, **kwargs):
        raise NotImplementedError("Baseline loss function from loss.py is not available. Please ensure loss.py exists and is importable.")

def calculate_unsupervised_tsp_loss(heatmap_output: torch.Tensor, coords: torch.Tensor,
                                    lambda_length: float, lambda_degree: float, lambda_subtour: float) -> torch.Tensor:
    batch_size, num_nodes, _ = heatmap_output.shape

    diagonal_mask = (1 - torch.eye(num_nodes, device=heatmap_output.device)).unsqueeze(0)
    heatmap_no_diag = heatmap_output * diagonal_mask

    distances = torch.cdist(coords, coords, p=2)
    L_length_raw = torch.sum(heatmap_no_diag * distances, dim=(-1, -2))
    L_length = (L_length_raw / (2.0 * num_nodes)).mean()

    node_degrees = torch.sum(heatmap_no_diag, dim=-1)
    L_degree = torch.mean(torch.sum((node_degrees - 2.0) ** 2, dim=-1))

    L_subtour_components = []
    for b in range(batch_size):
        adj_matrix = heatmap_no_diag[b]
        degree_vals = torch.sum(adj_matrix, dim=-1)
        degree_matrix = torch.diag(degree_vals)
        laplacian = degree_matrix - adj_matrix
        try:
            eigenvalues = torch.linalg.eigh(laplacian).eigenvalues
            if len(eigenvalues) < 2:
                 L_subtour_components.append(torch.tensor(0.0, device=heatmap_output.device))
                 continue
            fiedler_value = eigenvalues[1]
        except torch._C._LinAlgError as e:
            print(f"Warning: Eigenvalue computation failed for a batch item: {e}. Assigning high subtour penalty.")
            L_subtour_components.append(torch.tensor(float(num_nodes), device=heatmap_output.device))
            continue
        epsilon_fiedler = 1e-3
        L_subtour_component = torch.relu(epsilon_fiedler - fiedler_value) ** 2
        L_subtour_components.append(L_subtour_component)

    if not L_subtour_components:
        L_subtour = torch.tensor(0.0, device=heatmap_output.device)
    else:
        L_subtour = torch.stack(L_subtour_components).mean()

    # ---- START DEBUG PRINTS (Optional: uncomment for debugging specific runs) ----
    # if batch_idx < 2 and epoch ==1: # Example: print only for first 2 batches of first epoch
    #    print(f"  Epoch {epoch}, Batch {batch_idx} Debug Loss: L_length={L_length.item():.4f}, L_degree={L_degree.item():.4f}, L_subtour={L_subtour.item():.4f}")
    #    print(f"  Debug Heatmap: min={heatmap_no_diag.min().item():.4f}, max={heatmap_no_diag.max().item():.4f}, mean={heatmap_no_diag.mean().item():.4f}")
    #    print(f"  Debug Node Degrees: min={node_degrees.min().item():.4f}, max={node_degrees.max().item():.4f}, mean={node_degrees.mean().item():.4f}")
    # ---- END DEBUG PRINTS ----

    total_loss = (lambda_length * L_length +
                  lambda_degree * L_degree +
                  lambda_subtour * L_subtour)
    return total_loss

def train():
    parser = argparse.ArgumentParser(description="Train UTSP GNN model.")
    parser.add_argument('--num_of_nodes', type=int, default=20, help='Number of cities in TSP instances')
    parser.add_argument('--EPOCHS', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=5e-3, help='Learning rate')
    parser.add_argument('--stepsize', type=int, default=20, help='Learning rate decay step size')
    parser.add_argument('--gamma', type=float, default=0.1, help='Learning rate decay factor')
    parser.add_argument('--hidden', type=int, default=64, help='Hidden dimension of GNN layers')
    parser.add_argument('--nlayers', type=int, default=2, help='Number of GNN layers')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads in GNN')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate for GNN layers')
    parser.add_argument('--loss_type', type=str, default='alternative',
                        choices=['baseline', 'alternative'],
                        help='Type of loss function to use: "baseline" (original UTSP from loss.py) or "alternative" (new loss with Fiedler value).')
    parser.add_argument('--C1_penalty', type=float, default=20.0,
                        help='Penalty coefficient for L_cycle in the baseline UTSP loss (used if --loss_type baseline).')
    parser.add_argument('--lambda_length', type=float, default=1.0, help='Weight for tour length loss (alternative C_0)')
    parser.add_argument('--lambda_degree', type=float, default=50.0, help='Weight for degree-2 penalty (alternative C_1)')
    parser.add_argument('--lambda_subtour', type=float, default=20.0, help='Weight for subtour penalty (alternative C_2)')
    # --- NEW: Argument for gradient clipping ---
    parser.add_argument('--clip_grad_norm', type=float, default=1.0, help='Max norm for gradient clipping (0 to disable).')


    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = UTSPGNN(
        input_dim=2,
        hidden_dim=args.hidden,
        num_layers=args.nlayers,
        num_heads=args.num_heads,
        dropout=args.dropout
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)

    print(f"\n--- Starting Training ---")
    print(f"Nodes: {args.num_of_nodes}, Epochs: {args.EPOCHS}, Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}, LR Scheduler: StepLR (step={args.stepsize}, gamma={args.gamma})")
    print(f"Gradient Clipping Max Norm: {args.clip_grad_norm if args.clip_grad_norm > 0 else 'Disabled'}")
    print(f"Selected Loss Type: {args.loss_type.upper()}")

    if args.loss_type == 'baseline':
        if not BASELINE_LOSS_AVAILABLE:
            print("ERROR: Baseline loss was selected, but 'calculate_utsp_loss' could not be imported from loss.py.")
            return
        print(f"Baseline Loss Params: C1_penalty={args.C1_penalty}")
    elif args.loss_type == 'alternative':
        print(f"Alternative Loss Params: L_length={args.lambda_length}, L_degree={args.lambda_degree}, L_subtour={args.lambda_subtour}")

    print(f"Model Hidden Dim: {args.hidden}, Layers: {args.nlayers}, Heads: {args.num_heads}")
    print(f"---")

    num_batches_per_epoch = 100

    for epoch in range(1, args.EPOCHS + 1):
        model.train()
        total_epoch_loss = 0.0

        for batch_idx in range(num_batches_per_epoch):
            batch_coords = torch.stack([
                generate_tsp_instance(args.num_of_nodes) for _ in range(args.batch_size)
            ]).to(device)

            heatmaps = model(batch_coords)

            if args.loss_type == 'baseline':
                current_loss = baseline_calculate_utsp_loss(
                    heatmaps=heatmaps,
                    coords=batch_coords,
                    C1_penalty=args.C1_penalty
                )
            elif args.loss_type == 'alternative':
                # Pass batch_idx and epoch if you want to use them for conditional debug prints inside the loss function
                current_loss = calculate_unsupervised_tsp_loss(
                    heatmap_output=heatmaps,
                    coords=batch_coords,
                    lambda_length=args.lambda_length,
                    lambda_degree=args.lambda_degree,
                    lambda_subtour=args.lambda_subtour
                )
            else:
                raise ValueError(f"Unknown loss_type: {args.loss_type}. Choose 'baseline' or 'alternative'.")

            optimizer.zero_grad()
            current_loss.backward()
            # --- GRADIENT CLIPPING ADDED ---
            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_norm)
            optimizer.step()

            total_epoch_loss += current_loss.item()

        scheduler.step()
        avg_epoch_loss = total_epoch_loss / num_batches_per_epoch
        print(f"Epoch {epoch}/{args.EPOCHS}, Avg Loss: {avg_epoch_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

    print(f"\n--- Training Finished ---")
    model_save_path = f"utsp_model_N{args.num_of_nodes}_E{args.EPOCHS}_Loss_{args.loss_type}.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    train()