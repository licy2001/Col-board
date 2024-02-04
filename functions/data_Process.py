import torch


def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)


# def data_transform(config, X):
#     # 均匀去量化
#     if config.data.uniform_dequantization:
#         X = X / 256.0 * 255.0 + torch.rand_like(X) / 256.0
#     # 高斯去量化
#     if config.data.gaussian_dequantization:
#         X = X + torch.randn_like(X) * 0.01
#     # 重新缩放
#     if config.data.rescaled:
#         X = 2 * X - 1.0
#     # logit变换
#     elif config.data.logit_transform:
#         X = logit_transform(X)

#     if hasattr(config, "image_mean"):
#         return X - config.image_mean.to(X.device)[None, ...]

#     return X


# def inverse_data_transform(config, X):
#     if hasattr(config, "image_mean"):
#         X = X + config.image_mean.to(X.device)[None, ...]

#     if config.data.logit_transform:
#         X = torch.sigmoid(X)
#     elif config.data.rescaled:
#         X = (X + 1.0) / 2.0

#     return torch.clamp(X, 0.0, 1.0)


def data_transform(x):
    return 2 * x - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)
