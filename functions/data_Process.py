import torch


def data_transform(x):
    x = torch.clamp(x, 0, 1)
    return 2 * x - 1.0


def inverse_data_transform(x):
    x = torch.clamp(x, -1, 1)
    return torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)
