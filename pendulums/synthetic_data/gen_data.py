import sklearn.model_selection as model_selection
from torch.utils.data import DataLoader
import numpy as np
import torch


def get_synthetic_data(
    args,
    alpha=120.0,
    seed=0,
    ref_points=10,
    total_points=51,
    add_noise=True,
):
    np.random.seed(seed)
    ground_truth, ground_truth_tp = [], []
    observed_values = []
    for _ in range(args.n):
        key_values = np.random.randn(ref_points)
        key_points = np.linspace(0, 1, ref_points)
        query_points = np.linspace(0, 1, total_points)
        weights = np.exp(-alpha * (
            np.expand_dims(query_points, 1) - np.expand_dims(key_points, 0)
        ) ** 2)
        weights /= weights.sum(1, keepdims=True)
        query_values = np.dot(weights, key_values)
        ground_truth.append(query_values)
        if add_noise:
            noisy_query_values = query_values + 0.1 * np.random.randn(total_points)
        observed_values.append(noisy_query_values)
        ground_truth_tp.append(query_points)

    observed_values = np.array(observed_values)
    ground_truth = np.array(ground_truth)
    ground_truth_tp = np.array(ground_truth_tp)
    observed_mask = np.ones_like(observed_values)

    observed_values = np.concatenate(
        (
            np.expand_dims(observed_values, axis=2),
            np.expand_dims(observed_mask, axis=2),
            np.expand_dims(ground_truth_tp, axis=2),
        ),
        axis=2,
    )
    print(observed_values.shape)
    train_data, test_data = model_selection.train_test_split(
        observed_values, train_size=0.8, random_state=42, shuffle=True
    )
    _, ground_truth_test = model_selection.train_test_split(
        ground_truth, train_size=0.8, random_state=42, shuffle=True
    )
    _, val_data = model_selection.train_test_split(
        train_data, train_size=0.8, random_state=42, shuffle=True
    )
    print(train_data.shape, val_data.shape, test_data.shape)
    train = torch.from_numpy(train_data).float()
    val = torch.from_numpy(val_data).float()
    test = torch.from_numpy(test_data).float()

    data_objects = {
        "train": train,
        "test": test,
        "val": val,
        "input_dim": 1,
        "ground_truth": ground_truth_test,
    }
    return data_objects