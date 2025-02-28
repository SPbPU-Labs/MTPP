import multiprocessing
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import time
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from joblib import Parallel, delayed


# Load CIFAR-10 dataset of images
def load_imbalanced_data():
    (x_train, y_train), _ = cifar10.load_data()
    class_1, class_2 = 0, 1  # Select "airplane" and "automobile" class images

    mask_1 = y_train.flatten() == class_1
    mask_2 = y_train.flatten() == class_2
    x_class_1 = x_train[mask_1][:5000]  # |
    x_class_2 = x_train[mask_2][:500]   # | disbalanced dataset

    x_imbalanced = np.concatenate([x_class_1, x_class_2], axis=0)
    return x_imbalanced

# Data augmentation function
def augment_image(image):
    datagen = ImageDataGenerator(
        horizontal_flip=True,
        rotation_range=30,
        zoom_range=0.2
    )
    image = np.expand_dims(image, 0)  # Требуется 4D формат
    it = datagen.flow(image, batch_size=1)
    return next(it)[0].astype(np.uint8)

# Single process augmentation
def sequential_augmentation(images):
    return [augment_image(img) for img in images]

# Multiprocess augmentation
def parallel_augmentation(images, n_jobs):
    return Parallel(n_jobs=n_jobs)(delayed(augment_image)(img) for img in images)


if __name__ == '__main__':
    images = load_imbalanced_data()
    # Duplicate dataset to increase images variation
    images = [image for _ in range(20) for image in images]
    print(f"Images length: {len(images)}")

    # Sequential augmentation
    start_time = time.time()
    sequential_augmentation(images)
    single_core_time = time.time() - start_time
    print(f"Sequential time: {single_core_time:.2f} sec")

    # Parallel augmentation
    n_jobs = multiprocessing.cpu_count()
    start_time = time.time()
    parallel_augmentation(images, n_jobs=n_jobs)
    multi_core_time = time.time() - start_time
    print(f"Parallel time ({n_jobs} cores): {multi_core_time:.2f} sec")

    # Results output
    speedup = single_core_time / multi_core_time
    print(f"Speedup: {speedup:.2f}x")
