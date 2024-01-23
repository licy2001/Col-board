import os
import torch
import torchvision

from matplotlib import pyplot as plt
import pandas as pd
import logging

def save_image_list(image_list, output_path, format='png'):
    fig, axes = plt.subplots(1, len(image_list), figsize=(20, 4))

    for i in range(len(image_list)):
        img = image_list[i].to(torch.float32).cpu()

        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)

        img = img.permute(1, 2, 0).numpy()

        if img.shape[2] == 1:
            img = img.squeeze()
            cmap = 'gray'
        else:
            cmap = None

        axes[i].imshow(img, cmap=cmap)
        axes[i].axis('off')
        axes[i].set_title(f'{i}')

    plt.savefig(output_path, format=format)
    plt.clf()
    plt.close("all")
def save_image_dict(image_dict, output_path, type='png', facecolor='white', edgecolor='white', transparent=False):
    """
    mult_img_list 是一个包含了PyTorch张量图像的字典
    """
    # 创建一个7x1的子图布局
    fig, axes = plt.subplots(1, len(image_dict), figsize=(20, 4))
    for i, (img_name, img_tensor) in enumerate(image_dict.items()):
        # 如果图像在GPU上，先移动到CPU
        # img = img_tensor
        img = img_tensor.to(torch.float32).cpu()
        # 如果图像张量有批次维度，去掉批次维度
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)
        # 转换为NumPy数组，并确保通道在最后一个维度
        img = img.permute(1, 2, 0).numpy()
        # 如果图像是灰度图（即只有一个通道），使用灰度色彩映射
        if img.shape[2] == 1:
            img = img.squeeze()
            cmap = 'gray'
        else:
            cmap = None
            # 显示图像
        axes[i].imshow(img, cmap=cmap)
        axes[i].axis('off')
        axes[i].set_title(img_name) # 设置标题
    plt.savefig(output_path, format=type, facecolor='white', edgecolor='white', transparent=False)
    plt.clf()
    plt.close("all")
    # plt.show()

# display_and_save_images(mult_img_list, 'mult_img_display.png', format='png')
# display_and_save_images(mult_img_list, 'mult_img_display.jpg', format='jpg')
# display_and_save_images(mult_img_list, 'mult_img_display.pdf', format='pdf')
def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False):
    '''
    set up logger
    '''
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s', datefmt='%y-%m-%d %H:%M:%S')
    log_file = os.path.join(root, '{}.log'.format(phase))
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        l.addHandler(sh)


def loss_plot(data_list, path, x_label="Epoch", y_label="Loss"):
    plt.figure()
    # Plot and save the loss curve
    plt.plot(range(len(data_list)), data_list, linewidth=2, label=y_label)
    plt.grid(True)

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.title(f"{y_label} of {x_label}")
    plt.legend(loc="upper right")
    save_loss_path = path
    plt.savefig(save_loss_path, facecolor='white', edgecolor='white', transparent=False)
    plt.clf()
    plt.close("all")


def loss_table(data_list, path, y1_label, y2_label):
    # 创建一个 DataFrame
    df = pd.DataFrame({y1_label: range(0, len(data_list)), y2_label: data_list})
    df[y2_label] = df[y2_label].round(6)
    save_table_path = path
    # 保存为 xls 文件
    df.to_excel(save_table_path, index=False)


def save_image(img, file_directory):
    if not os.path.exists(os.path.dirname(file_directory)):
        os.makedirs(os.path.dirname(file_directory))
    torchvision.utils.save_image(img, file_directory)


def save_checkpoint(state, filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    torch.save(state, filename + ".pth")


def get_network_description(network):
    if isinstance(network, torch.nn.DataParallel):
        network = network.module
    s = str(network)
    n = sum(map(lambda x: x.numel(), network.parameters()))
    return s, n


def load_checkpoint(path, device):
    if device is None:
        return torch.load(path, map_location=torch.device("cpu"))
    else:
        return torch.load(path, map_location=torch.device("cpu"))
