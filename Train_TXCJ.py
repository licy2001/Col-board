import sys
import argparse
import os
import yaml
import torch
import numpy as np
from models.ddpm import DDPM
from functions.get_Dataset import Restruction


def parse_args_and_config():
    parser = argparse.ArgumentParser(
        description="Training Denoising Diffusion Models  For TXCJ"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="/data2/wait/bisheCode/DDPM_Fusion/config/coco.yml",
        help="Path to the config file",
    )
    parser.add_argument("--phase", type=str, default="train", help="val(generation)")
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        help="Path for checkpoint to load and resume",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=20,
        help="Number of implicit sampling steps for validation image patches",
    )
    parser.add_argument(
        "--seed",
        default=61,
        type=int,
        metavar="N",
        help="Seed for initializing training (default: 61)",
    )
    parser.add_argument(
        "-i",
        "--image_folder",
        type=str,
        default="/data2/wait/bisheCode/DDPM_Fusion/images",
        help="The folder name of samples",
    )
    parser.add_argument("-gpu", "--gpu_ids", type=str, default="1")
    parser.add_argument(
        "--name",
        type=str,
        default="TXCJ",
        help="folder name to save outputs",
    )
    args = parser.parse_args()

    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)
    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()
    # setup device to run
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device: {}".format(device))
    config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    # data loading
    print("=> using dataset '{}'".format(config.data.dataset))
    DATASET = Restruction(config)

    # create model
    print("=> creating denoising-diffusion model...")
    diffusion = DDPM(args, config)
    diffusion.train(DATASET)


if __name__ == "__main__":
    main()