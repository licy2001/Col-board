import torch
import torch.nn.functional as F


def gradient(input_tensor, direction):
    """
    计算梯度
    smooth_kernel_x/y是一个用于计算x/y方向梯度的平滑核，它的作用是检测图像中的平滑变化，通常用于边缘检测和纹理分析。
    """
    # Calculate gradient using classical Sobel filter
    sobel_filter_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(
        input_tensor.device)
    sobel_filter_y = sobel_filter_x.transpose(2, 3)
    # smooth_kernel_x = torch.tensor([[0, 0], [-1, 1]], dtype=torch.float32).view(1, 1, 2, 2)
    # smooth_kernel_y = smooth_kernel_x.transpose(2, 3)

    if direction == "x":
        kernel = sobel_filter_x
    elif direction == "y":
        kernel = sobel_filter_y
    else:
        raise ValueError("Invalid direction. Use 'x' or 'y'.")

    # Apply convolution
    # gradient_orig = torch.abs(F.conv2d(input_tensor, kernel, stride=1, padding=1))
    gradient_tensor = F.conv2d(input_tensor, kernel, padding=1)

    # # Normalize gradient values
    # grad_min = torch.min(gradient_orig)
    # grad_max = torch.max(gradient_orig)
    # grad_norm = (gradient_orig - grad_min) / (grad_max - grad_min + 0.0001)

    return gradient_tensor


def gradient_loss(img1, img2):
    """
    梯度损失
    计算图像的梯度 gradient_img1_x是一个用于计算水平方向梯度的卷积核，它的作用是检测图像中的水平方向的变化，通常用于边缘检测和特征提
    """
    gradient_img1_x = F.conv2d(img1,
                               torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3),
                               stride=1, padding=1)
    gradient_img1_y = F.conv2d(img1,
                               torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3),
                               stride=1, padding=1)

    gradient_img2_x = F.conv2d(img2,
                               torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3),
                               stride=1, padding=1)
    gradient_img2_y = F.conv2d(img2,
                               torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3),
                               stride=1, padding=1)

    # 计算梯度损失
    gradient_loss_x = torch.mean(torch.abs(gradient_img1_x - gradient_img2_x))
    gradient_loss_y = torch.mean(torch.abs(gradient_img1_y - gradient_img2_y))

    return gradient_loss_x + gradient_loss_y


def content_loss(img1, img2):
    # 计算内容损失，例如使用均方误差（MSE）
    return F.mse_loss(img1, img2, reduction="mean")


def total_loss(img_gt, img_generated):
    # 添加梯度损失、内容损失等
    alpha = 0.1  # 调整权重
    loss_gradient = gradient_loss(img_gt, img_generated)
    loss_content = content_loss(img_gt, img_generated)

    # 总体损失函数
    loss_total = (1 - alpha) * loss_content + alpha * loss_gradient

    return loss_total
