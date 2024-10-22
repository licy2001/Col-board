import argparse
import os
import yaml
import torch
import numpy as np
from functions.get_Dataset import Fusion
from models.ddpm import DDPM


def parse_args_and_config():
    parser = argparse.ArgumentParser(
        description="Restoring Weather with Patch-Based Denoising Diffusion Models"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="/data2/wait/bisheCode/Fusion/config/TXCJ64.yml",
        help="Path to the config file",
    )
    parser.add_argument(
        "--resume",
        default="/data2/wait/bisheCode/Fusion/results/TXCJ64/checkpoint/TXCJ_epoch_2100.pth",
        type=str,
        help="Path for the diffusion model checkpoint to load for evaluation",
    )
    parser.add_argument(
        "--data",
        default="/data2/wait/bisheCode/Fusion/dataset/TNO",
        type=str,
        help="Path for the fusion data",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=25,
        help="10 Number of implicit sampling steps",
    )

    parser.add_argument("-gpu", "--gpu_ids", type=str, default="0")
    parser.add_argument(
        "--seed",
        default=61,  # 61
        type=int,
        metavar="N",
        help="Seed for initializing training (default: 61)",
    )
    parser.add_argument(
        "--concat_type",
        type=str,
        default="ABX",
        help="the concat type of condition Image",
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
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    # setup device to run
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: {}".format(device))
    config.device = device

    if torch.cuda.is_available():
        print(
            "Note: Currently supports evaluations (restoration) when run only on a single GPU!"
        )

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    # data loading
    print("=> using dataset '{}'".format(args.data))
    DATASET = Fusion(config)

    dataloader = DATASET.get_fusion_loaders(
        parent_dir=args.data,
        batch_size=1,
    )
    # create model
    print("=> creating denoising-diffusion model with wrapper...")
    diffusion = DDPM(args, config)
    diffusion.Fusion_sample(dataloader, type="TNO")


if __name__ == "__main__":
    main()
