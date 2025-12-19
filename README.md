# **Unsupervised GNN Solver for the Travelling Salesman Problem**

## **Project Overview**

This project provides a PyTorch implementation for solving the Travelling Salesman Problem (TSP) using an unsupervised learning framework with Graph Neural Networks (GNNs). The approach is based on the work of Joshi et al. in "An Unsupervised Learning Framework for Combinatorial Optimization with Applications to the Traveling Salesman Problem" (UTSP).

This implementation replicates and extends the original framework by:

1. Using a **Graph Attention Network (GAT)** as the core GNN architecture to learn inter-city relationships.  
2. Designing and evaluating a novel **alternative surrogate loss function** that incorporates the Fiedler value of the graph Laplacian to explicitly enforce tour connectivity.  
3. Conducting a rigorous experimental comparison between the baseline UTSP loss and the alternative loss across multiple problem sizes (N=20, 50, 100, 200, and 500).

The model is trained to predict a heatmap of edge probabilities, from which a final tour is constructed using a greedy search followed by 2-Opt local search refinement.

## **Repository Structure**

.  
├── utsp\_project/  
│   ├── \_\_init\_\_.py  
│   ├── dataset.py  
│   ├── loss.py  
│   ├── model.py  
│   └── tsp\_env.py  
├── train.py  
├── loadmodel.py  
├── Create\_Charts.py  
├── experiments/  
├── Results/  
├── README.md  
└── requirements.txt

* utsp\_project/: Core Python package containing the UTSP implementation  
  - model.py: Contains the GAT-based GNN model architecture.  
  - loss.py: Implements the baseline UTSP surrogate loss function.  
  - dataset.py: Utility for generating random Euclidean TSP instances.  
  - tsp\_env.py: Contains the logic for tour construction (greedy search) and 2-Opt local search refinement, optimized with Numba.  
* train.py: The main script for training models. Supports both the baseline and the alternative loss function.  
* loadmodel.py: The script for evaluating trained models on new TSP instances.  
* Create\_Charts.py: Script for generating result visualizations.  
* experiments/: Contains baseline comparison implementations.  
* Results/: Stores trained models, result charts, and analysis documents.  
* requirements.txt: Lists the required Python libraries.

## **Setup**

1. **Clone the repository:**  
   git clone \[your-repo-url\]  
   cd \[your-repo-name\]

2. **Create a virtual environment (recommended):**  
   python \-m venv venv  
   source venv/bin/activate  \# On Windows, use \`venv\\Scripts\\activate\`

3. **Install dependencies:**  
   pip install \-r requirements.txt

## **Usage**

### **Training Models**

The train.py script is used to train models. For larger problem sizes (N\>=100), training on a CUDA-enabled GPU is strongly recommended.

#### **Training with the Alternative Loss (Our Modification)**

Use \--loss\_type alternative and specify the lambda weights.

\# Example for N=100 (Best performing alternative settings)  
python train.py \--num\_of\_nodes 100 \--EPOCHS 120 \--batch\_size 4 \\  
\--loss\_type alternative \\  
\--lambda\_length 1.0 \--lambda\_degree 50.0 \--lambda\_subtour 40.0 \\  
\--lr 0.0005 \--clip\_grad\_norm 1.0 \--stepsize 20 \--gamma 0.3

\# Example for N=500  
python train.py \--num\_of\_nodes 500 \--EPOCHS 250 \--batch\_size 1 \\  
\--loss\_type alternative \\  
\--lambda\_length 1.0 \--lambda\_degree 50.0 \--lambda\_subtour 45.0 \\  
\--lr 0.0003 \--clip\_grad\_norm 1.0 \--stepsize 35 \--gamma 0.3

#### **Training with the Baseline UTSP Loss**

Use \--loss\_type baseline and specify the C1\_penalty.

\# Example for N=200 (Best performing baseline settings)  
python train.py \--num\_of\_nodes 200 \--EPOCHS 150 \--batch\_size 2 \\  
\--loss\_type baseline \--C1\_penalty 20.0 \\  
\--lr 0.0005 \--clip\_grad\_norm 1.0 \--stepsize 20 \--gamma 0.3

\# Example for N=500  
python train.py \--num\_of\_nodes 500 \--EPOCHS 250 \--batch\_size 1 \\  
\--loss\_type baseline \--C1\_penalty 20.0 \\  
\--lr 0.0001 \--clip\_grad\_norm 1.0 \--stepsize 35 \--gamma 0.3

*(Note: For training on Google Colab, you can add \--save\_dir "/path/to/your/gdrive" to save models directly to Google Drive.)*

### **Evaluating Models**

The loadmodel.py script is used to evaluate trained models on new, randomly generated instances.

\# Example: Evaluate the N=100 alternative model trained for 120 epochs  
python loadmodel.py \--num\_of\_nodes 100 \--epochs\_trained 120 \--loss\_type\_trained alternative

\# Example: Evaluate the N=500 baseline model trained for 250 epochs  
\# (Using fewer test instances due to evaluation time)  
python loadmodel.py \--num\_of\_nodes 500 \--epochs\_trained 250 \--loss\_type\_trained baseline \--num\_test\_instances 20

*(Note: If evaluating a model trained on Colab, add \--load\_dir "/path/to/your/gdrive" or place the .pth model file in the local directory.)*

### **Key Script Arguments**

* \--num\_of\_nodes: Number of cities in the TSP instance (e.g., 20, 50, 100, 200, 500).  
* \--EPOCHS: Number of training epochs.  
* \--batch\_size: Number of instances per training batch (use 1 or 2 for large N).  
* \--loss\_type: baseline or alternative.  
* \--C1\_penalty: Weight for the cycle constraint in the baseline loss.  
* \--lambda\_\*: Weights for the different components of the alternative loss.  
* \--lr: Initial learning rate.  
* \--clip\_grad\_norm: Max norm for gradient clipping (e.g., 1.0). Use 0 to disable.  
* \--stepsize, \--gamma: Parameters for the StepLR learning rate scheduler.  
* \--save\_dir: (Optional) Directory to save trained models.  
* \--load\_dir: (Optional) Directory to load trained models from for evaluation.
