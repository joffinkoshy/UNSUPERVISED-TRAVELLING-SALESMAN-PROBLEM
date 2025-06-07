# loadmodel.py
import torch
import numpy as np
import argparse
import time  # <--- MOVED IMPORT TIME HERE

from model import UTSPGNN  # Your GNN model
from dataset import generate_tsp_instance  # For generating test coordinates
# Ensure tsp_env.py and its functions are available
from tsp_env import generate_tour_from_heatmap_and_coords


def evaluate_model():
    parser = argparse.ArgumentParser(description="Evaluate a trained UTSP GNN model.")
    # Arguments for loading the model and evaluation setup
    parser.add_argument('--num_of_nodes', type=int, default=20,  # Example: 20, 50
                        help='Number of cities in TSP instances to evaluate')
    parser.add_argument('--epochs_trained', type=int, default=10,  # Example: matching EPOCHS from training
                        help='Number of epochs the model was trained for (to find saved file)')

    # --- NEW: Argument to specify the loss type of the saved model ---
    parser.add_argument('--loss_type_trained', type=str, default='alternative',
                        choices=['baseline', 'alternative'],
                        help='Loss type the model was trained with: "baseline" or "alternative". This affects the filename.')

    parser.add_argument('--num_test_instances', type=int, default=100,
                        help='Number of new instances to evaluate on')

    # Model Hyperparameters (MUST match what the model was trained with)
    # These should match the parameters used during the training of the specific model being loaded.
    parser.add_argument('--hidden', type=int, default=64, help='Hidden dimension of GNN layers')
    parser.add_argument('--nlayers', type=int, default=2, help='Number of GNN layers')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads in GNN')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate for GNN layers')

    args = parser.parse_args()

    # --- UPDATED: Construct model_path using the new loss_type_trained argument ---
    model_path = f"utsp_model_N{args.num_of_nodes}_E{args.epochs_trained}_Loss_{args.loss_type_trained}.pth"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Starting Evaluation ---")
    print(f"Using device for evaluation: {device}")
    print(
        f"Loading model for N={args.num_of_nodes}, Epochs Trained={args.epochs_trained}, Loss Type={args.loss_type_trained.upper()}")
    print(f"Model path: {model_path}")

    # 1. Instantiate the Model with the correct architecture
    # Ensure these hyperparameters match those used for training the specific model file being loaded.
    model = UTSPGNN(
        input_dim=2,  # x,y coordinates
        hidden_dim=args.hidden,
        num_layers=args.nlayers,
        num_heads=args.num_heads,
        dropout=args.dropout  # Dropout is typically turned off by model.eval(), but good to match architecture
    ).to(device)

    # 2. Load the Trained Model State
    try:
        # Consider setting weights_only=True for security if loading untrusted models
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))  # Or True
        print(f"Successfully loaded model from {model_path}")
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {model_path}.")
        print(
            f"Please check the path and ensure that 'num_of_nodes', 'epochs_trained', and 'loss_type_trained' arguments match a saved model file.")
        return
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        print(
            f"This can happen if the model architecture defined here (hyperparameters like hidden, nlayers, num_heads) does not match the architecture of the saved model.")
        return

    model.eval()  # Set model to evaluation mode (CRUCIAL: disables dropout, uses moving averages for BatchNorm if any)

    all_tour_lengths = []
    all_heatmap_times = []
    all_localsearch_times = []  # To store local search times

    print(f"\n--- Evaluating on {args.num_test_instances} new instances (N={args.num_of_nodes}) ---")
    for i in range(args.num_test_instances):
        # Generate a new Test TSP Instance
        # Adding a batch dimension [1, num_nodes, 2] as the model expects a batch
        coords_batch = generate_tsp_instance(args.num_of_nodes, device=device).unsqueeze(0)
        coords_np = coords_batch.squeeze(0).cpu().numpy()  # For local search and distance calculation

        # Infer Heatmaps
        # Corrected timing logic:
        heatmap_time_s = 0.0
        if device.type == 'cuda':
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            cpu_start_time = time.time()

        with torch.no_grad():  # Disable gradient computation during inference for efficiency
            predicted_heatmap_batch = model(coords_batch)  # [1, num_nodes, num_nodes]

        if device.type == 'cuda':
            end_event.record()
            torch.cuda.synchronize()  # Wait for the event to complete
            heatmap_time_s = start_event.elapsed_time(end_event) / 1000.0  # Convert ms to s
        else:
            heatmap_time_s = time.time() - cpu_start_time
        all_heatmap_times.append(heatmap_time_s)

        predicted_heatmap_np = predicted_heatmap_batch.squeeze(0).cpu().numpy()  # Remove batch dim, to NumPy

        # Extract Tours from Heatmaps (Decoding) and apply local search
        try:
            ls_start_time = time.time()  # Local search is CPU-bound with NumPy/Numba
            tour_nodes_sequence, tour_length = generate_tour_from_heatmap_and_coords(
                predicted_heatmap_np,
                coords_np
            )
            localsearch_time_s = time.time() - ls_start_time
            all_localsearch_times.append(localsearch_time_s)

            if tour_length == float('inf'):
                print(f"Warning: Instance {i + 1} resulted in a failed tour extraction (infinite length). Skipping.")
                continue  # Skip this instance if tour extraction failed

            all_tour_lengths.append(tour_length)

        except Exception as e:
            print(f"Warning: Could not extract tour for instance {i + 1}. Error: {e}")
            # You might want to skip this instance or count failed extractions
            continue

        if (args.num_test_instances <= 20) or ((i + 1) % (
        args.num_test_instances // 10 if args.num_test_instances > 10 else 1) == 0):  # Adjusted progress print condition
            print(f"Processed {i + 1}/{args.num_test_instances} instances. Last tour length: {tour_length:.4f}")

    # Calculate Metrics
    if all_tour_lengths:
        avg_tour_length = np.mean(all_tour_lengths)
        std_tour_length = np.std(all_tour_lengths)
        avg_heatmap_time = np.mean(all_heatmap_times)
        avg_localsearch_time = np.mean(all_localsearch_times)

        print(f"\n--- Evaluation Results (N={args.num_of_nodes}) ---")
        print(f"Total instances successfully evaluated: {len(all_tour_lengths)}/{args.num_test_instances}")
        print(f"Average Tour Length: {avg_tour_length:.4f}")
        print(f"Standard Deviation of Tour Lengths: {std_tour_length:.4f}")
        print(f"Average Heatmap Generation Time: {avg_heatmap_time:.6f} seconds")
        print(f"Average Local Search (2-Opt) Time: {avg_localsearch_time:.6f} seconds")
        print(f"Average Total Inference Time (Heatmap + LS): {(avg_heatmap_time + avg_localsearch_time):.6f} seconds")

        # Optional: Compare to Optimal - Add your optimal length data here if available
        # Example:
        # optimal_lengths_data = { 20: 5.67, 50: 10.23 } # Load appropriate optimal values for your generated instances
        # if args.num_of_nodes in optimal_lengths_data:
        #     optimal_avg_length = optimal_lengths_data[args.num_of_nodes]
        #     avg_optimality_gap = ((avg_tour_length - optimal_avg_length) / optimal_avg_length) * 100
        #     print(f"Average Optimality Gap from {optimal_avg_length:.4f}: {avg_optimality_gap:.2f}%")
        # else:
        #     print(f"No optimal solution data available for N={args.num_of_nodes} to calculate gap.")

    else:
        print("No tours were successfully extracted and evaluated.")
    print(f"--- Evaluation Finished ---")


if __name__ == "__main__":
    # The 'import time' is now at the top of the file, so it's available globally.
    evaluate_model()