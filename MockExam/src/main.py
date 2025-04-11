import concurrent.futures
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from utils import load_data, create_imbalanced_subsets, plot_distribution, create_model, balance_classes_with_augmentation


class FederatedLearning:
    def __init__(self, num_clients: int = 6, epochs: int = 2, rounds: int = 5, target_size: int = 1000):
        self.num_clients = num_clients
        self.epochs = epochs
        self.rounds = rounds
        self.global_model = create_model()
        (self.x_train, self.y_train), (self.x_test, self.y_test) = load_data()
        self.imbalanced_subsets = create_imbalanced_subsets(self.x_train, self.y_train, num_clients)
        self.subsets = balance_classes_with_augmentation(self.imbalanced_subsets, target_size)

    def train_client(self, client_data: Tuple[np.ndarray, np.ndarray]) -> List[np.ndarray]:
        model = create_model()
        model.set_weights(self.global_model.get_weights())
        x, y = client_data
        model.fit(x, y, epochs=self.epochs, verbose=0)
        return model.get_weights()

    def aggregate_weights(self, weights_list: List[List[np.ndarray]]) -> None:
        avg_weights = [
            np.mean(layer_weights, axis=0)
            for layer_weights in zip(*weights_list)
        ]
        self.global_model.set_weights(avg_weights)

    def evaluate_global_model(self) -> float:
        loss, acc = self.global_model.evaluate(self.x_test, self.y_test, verbose=0)
        return acc

    def run(self):
        plot_distribution(self.imbalanced_subsets, "plots/imbalanced_subset.png")
        plot_distribution(self.subsets, "plots/augmented_subset.png")
        print("Starting federated training...")
        for round in range(self.rounds):
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = [
                    executor.submit(self.train_client, subset)
                    for subset in self.subsets
                ]
                weights_list = [f.result() for f in tqdm(futures, desc=f"Round {round+1}")]
            self.aggregate_weights(weights_list)
            acc = self.evaluate_global_model()
            print(f"Round {round+1}: Global accuracy = {acc:.4f}")

def centralized_baseline():
    (x_train, y_train), (x_test, y_test) = load_data()
    model = create_model()
    model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.2)
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Centralized model accuracy: {acc:.4f}")

if __name__ == "__main__":
    import matplotlib
    os.makedirs("plots", exist_ok=True)
    matplotlib.use('Agg')  # No GUI, server plotting
    fl = FederatedLearning()
    fl.run()
    centralized_baseline()