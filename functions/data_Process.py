import torch


def data_transform(x):
    return 2 * x - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)
