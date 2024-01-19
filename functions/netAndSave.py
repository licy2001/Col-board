import os
import torch
import torchvision

from matplotlib import pyplot as plt
import pandas as pd


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
    plt.savefig(save_loss_path)
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
