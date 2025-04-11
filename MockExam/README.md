# Federated Learning with MNIST Data Imbalance

## Task

To understand the impact of data imbalance in federated learning by
simulating a distributed environment with varying data distributions.

Steps:

1. Dataset Loading:
Use the `keras.datasets.mnist.load_data()` function to load the MNIST dataset.
Explain the structure of the MNIST dataset (images and labels).
2. Data Partitioning (Creating Subsets):
Implement a function to partition the MNIST dataset into 6 distinct subsets randomly.
Visualize the distribution of digits in each subset using histograms or bar charts.
3. Federated Learning Simulation:
Simulate a federated learning environment where each subset represents the local data of a client.
4. Implement a federated averaging algorithm:  
Train the model locally on each client's subset for a few epochs.
Average the model weights from all clients to create a global model.
Repeat the local training and averaging process for multiple rounds.
5. Define a simple Keras Sequential model (e.g., a small CNN or a dense
   network) to be used for training on the entire dataset.
6. Evaluate the centralized model's performance on the same test set used in the
   federated learning simulation.

## Run project

```bash
pip install -r requirements.txt
python src/main.py
```
