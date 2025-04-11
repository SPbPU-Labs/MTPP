from concurrent.futures import ThreadPoolExecutor

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data() -> Tuple[Tuple, Tuple]:
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    return (x_train, y_train), (x_test, y_test)

def create_imbalanced_subsets(
        x: np.ndarray,
        y: np.ndarray,
        num_clients: int = 6,
        min_samples: int = 10,
        max_samples: int = 1000
) -> List[Tuple[np.ndarray, np.ndarray]]:
    subsets = []
    for _ in range(num_clients):
        indices = []
        for digit in range(10):
            digit_indices = np.where(y == digit)[0]
            n_samples = np.random.randint(min_samples, max_samples)
            selected = np.random.choice(digit_indices, size=min(n_samples, len(digit_indices)), replace=False)
            indices.extend(selected)
        np.random.shuffle(indices)
        subsets.append((x[indices], y[indices]))
    return subsets

def balance_classes_with_augmentation(
        subsets: List[Tuple[np.ndarray, np.ndarray]],
        samples_per_class: int = 1000  # Фиксированное число примеров для каждой цифры
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Выравнивает количество примеров для каждой цифры через аугментацию."""
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1
    )

    def _balance_client(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        balanced_x, balanced_y = [], []

        for digit in range(10):
            digit_indices = np.where(y == digit)[0]
            existing_samples = x[digit_indices]

            if len(existing_samples) < samples_per_class:
                needed = samples_per_class - len(existing_samples)
                augmented = []
                for batch in datagen.flow(existing_samples, batch_size=len(existing_samples), shuffle=False):
                    augmented.append(batch)
                    if len(augmented) * len(existing_samples) >= needed:
                        break
                augmented = np.concatenate(augmented)[:needed]
                existing_samples = np.concatenate([existing_samples, augmented])

            balanced_x.append(existing_samples[:samples_per_class])
            balanced_y.append(np.full(samples_per_class, digit))

        return np.concatenate(balanced_x), np.concatenate(balanced_y)

    with ThreadPoolExecutor() as executor:
        balanced_subsets = list(executor.map(_balance_client, [x for x, _ in subsets], [y for _, y in subsets]))

    return balanced_subsets

def plot_distribution(subsets: List, save_path: str = "plots/distributions.png"):
    plt.figure(figsize=(15, 8))
    for i, (_, y) in enumerate(subsets):
        plt.subplot(2, 3, i+1)
        plt.hist(y, bins=10, range=(0, 10))
        plt.title(f"Client {i+1}")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def create_model() -> keras.Model:
    model = keras.Sequential([
        keras.layers.Input(shape=(28, 28, 1)),
        keras.layers.Reshape((784,)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation="softmax")
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model
