import os, sys

sys.path.append("/data2/wait/bisheCode/DDPM_Fusion")
import logging
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
from torchvision.utils import save_image as save_img
import torch.backends.cudnn as cudnn
from models.unet import UNet
from models.ema import EMAHelper
from functions.netAndSave import (
    load_checkpoint,
    save_image,
    save_checkpoint,
    get_network_description,
    loss_plot,
    loss_table,
    setup_logger,
    save_image_dict,
)
from functions.get_Optimizer import get_optimizer
from functions.losses import noise_estimation_loss
from functions.data_Process import data_transform, inverse_data_transform
from functions import metrics


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    if beta_schedule == "quad":
        betas = (torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=torch.float64) ** 2)
    elif beta_schedule == "linear":
        betas = torch.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=torch.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * torch.ones(num_diffusion_timesteps, dtype=torch.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = torch.linspace(-6, 6, num_diffusion_timesteps)
        betas = torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class DDPM(object):
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.device = config.device
        self.model = UNet(config)
        self.model.to(self.device)
        gpus = [1]
        self.model = torch.nn.DataParallel(self.model)
        self.ema_helper = EMAHelper(mu=self.config.model.ema_rate)
        self.ema_helper.register(self.model)

        self.optimizer = get_optimizer(self.config, self.model.parameters())
        self.start_epoch, self.step = 0, 0
        self.epochs_loss, self.psnr, self.ssim = [], [], []

        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )

        # 参数
        self.betas = betas
        self.num_timesteps = betas.shape[0]
        self.alphas = 1.0 - betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)
        self.alphas_bar_prev = F.pad(self.alphas_bar[:-1], (1, 0), value=1.0)
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1.0 - self.alphas_bar)
        self.log_one_minus_alphas_bar = torch.log(1.0 - self.alphas_bar)
        self.sqrt_recip_alphas_bar = torch.sqrt(1.0 / self.alphas_bar)
        self.sqrt_recipm1_alphas_bar = torch.sqrt(1.0 / self.alphas_bar - 1)
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.p_variance = (
                self.betas * (1.0 - self.alphas_bar_prev) / (1.0 - self.alphas_bar)
        )
        self.p_log_variance_clipped = torch.log(self.p_variance.clamp(min=1e-20))

        # mu_theta_t 的系数
        self.p_mean_coef1 = self.sqrt_alphas_bar * (1 - self.alphas_bar_prev) / (1 - self.alphas_bar)
        self.p_mean_coef2 = torch.sqrt(self.alphas_bar_prev) * self.betas / (1 - self.alphas_bar)

        # setup:初始化操作
        self.setup(self.args, self.config)
        s, n = get_network_description(self.model)
        net_struc_str = "{}".format(self.model.__class__.__name__)
        self.logger.info(
            "Network G structure: {}, with parameters: {:,d}".format(net_struc_str, n)
        )

    # Get the param of given timestep t
    def extract(self, a: torch.Tensor, t: torch.Tensor):
        batch_size = t.shape[0]
        out = a.to(self.device).gather(0, t.long()).float()
        out = out.reshape(batch_size, 1, 1, 1)
        return out

    def load_ddm_ckpt(self, load_path, ema=False, train=True):
        print("checkpoint path:", load_path)
        checkpoint = load_checkpoint(load_path, None)
        self.start_epoch = checkpoint["epoch"]
        self.step = checkpoint["step"]
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        if train:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.ema_helper.load_state_dict(checkpoint["ema_helper"])
        if ema:
            self.ema_helper.ema(self.model)
            self.ema_helper.register(self.model)
        self.logger.info(
            "=> loaded checkpoint '{}' (epoch {}, step {})".format(
                load_path, self.start_epoch, self.step
            )
        )

    def setup(self, args, config):
        self.root_dir = os.path.join(config.path.root, "results", "{}".format(config.data.dataset))
        os.makedirs(self.root_dir, exist_ok=True)
        self.log_dir = os.path.join(self.root_dir, config.path.log)
        os.makedirs(self.log_dir, exist_ok=True)
        self.figure_dir = os.path.join(self.root_dir, config.path.figure)
        self.generate_process_dir = os.path.join(self.root_dir, config.path.train_process)
        self.val_sample_dir = os.path.join(self.root_dir, config.path.val_sample)
        self.test_sample_dir = os.path.join(self.root_dir, config.path.test_sample)
        self.checkpoint_dir = os.path.join(self.root_dir, config.path.checkpoint)
        self.fusion_dir = os.path.join(config.path.root, config.path.fusion)

        setup_logger(None, self.log_dir, "train", level=logging.INFO, screen=True)
        setup_logger("val", self.log_dir, "val", level=logging.INFO)
        self.logger = logging.getLogger("base")
        self.logger_val = logging.getLogger("val")

    # 生成时间步
    def sample_timestep(self, n, symmetric=True, big_range=True):
        if symmetric:
            if big_range:
                # 生成整数的范围更大
                t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
            else:
                t = torch.randint(low=0, high=self.num_timesteps // 2, size=(n // 2 + 1,)).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
        else:
            t = torch.randint(low=0, high=self.num_timesteps, size=(n,)).to(self.device)  # 生成n个随机时间步骤
        return t

    def get_loss(self, x0, x_cond):
        """
        计算MSE损失
        """
        t = self.sample_timestep(x0.shape[0], symmetric=False, big_range=True)
        eps = torch.randn_like(x0).to(self.device)
        xt = self.sample_forward(x0, t, eps)
        if self.args.concat_type == "ABX":
            model_input = torch.cat([x_cond, xt], dim=1)
        elif self.args.concat_type == "AXB":
            model_input = torch.cat([x_cond[:, :3, :, :], xt, x_cond[:, 3:, :, :]], dim=1)
        else:
            model_input = torch.cat([xt, x_cond], dim=1)

        eps_theta = self.model(model_input, t)
        x0_coef1 = self.extract(self.sqrt_recip_alphas_bar, t)
        x0_coef2 = self.extract(self.sqrt_one_minus_alphas_bar, t)
        pred_x0 = x0_coef1 * (xt - x0_coef2 * eps_theta)
        pred_x0 = torch.clamp(pred_x0, -1, 1)
        loss_fn = torch.nn.MSELoss(size_average=False, reduce=True, reduction="sum")
        loss = loss_fn(eps_theta, eps)
        return loss, xt, eps, eps_theta, pred_x0

    def train(self, DATASET):
        cudnn.benchmark = True
        train_loader, val_loader = DATASET.get_loaders()
        if os.path.isfile(self.args.resume):
            self.load_ddm_ckpt(self.args.resume, ema=True, train=True)

        best_psnr = 0
        epochs_losses = self.epochs_loss
        psnr = self.psnr
        ssim = self.ssim

        print(self.start_epoch)
        for epoch in range(self.start_epoch, self.config.training.n_epochs):
            data_start = time.time()
            data_time = 0
            epoch_loss = 0.0
            epoch_start_time = time.time()
            for i, (x, y) in enumerate(tqdm(train_loader)):
                self.step += 1
                # print("input shape:", x.shape)
                # x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                # print("flatten input shape:", x.shape)
                current_batch_size = x.shape[0]
                n = current_batch_size
                data_time += time.time() - data_start
                # 开启梯度更新
                self.model.train()

                x = x.to(self.device)
                x = data_transform(x)
                # antithetic sampling
                x0 = x[:, 6:, :, :]
                x_cond = x[:, :6, :, :]
                # loss, x_t, pred_noise, pred_x0 = noise_estimation_loss(
                #     self.model, x0, x_cond, t, e, b
                # )
                loss, x_t, eps, pred_noise, pred_x0 = self.get_loss(x0=x0, x_cond=x_cond)
                self.logger.info(
                    f"step: {self.step}\tloss: {loss.item():.8f}\tdata time: {data_time / (i + 1):.4f}"
                )
                # 模型参数更新
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.config.model.ema:
                    self.ema_helper.update(self.model)

                # epoch_loss += loss.item() * current_batch_size
                epoch_loss += loss.item()
                if self.step % 100 == 0 or self.step == 1:
                    mult_img_dict = {
                        'cond1': inverse_data_transform(x[0, :3, :, :].detach().float().cpu()),
                        'cond2': inverse_data_transform(x[0, 3:6, :, :].detach().float().cpu()),
                        'xt': inverse_data_transform(x_t[0, ::].detach().float().cpu()),
                        'noise': inverse_data_transform(eps[0, ::].detach().float().cpu()),
                        'pred_noise': inverse_data_transform(pred_noise[0, ::].detach().float().cpu()),
                        'pred_x0': inverse_data_transform(pred_x0[0, ::].detach().float().cpu()),
                        'x0': inverse_data_transform(x[0, 6:, ::].detach().float().cpu())
                    }
                    os.makedirs(self.generate_process_dir, exist_ok=True)
                    mult_img_dict_path = os.path.join(self.generate_process_dir, f"grid_mult_img{self.step}.png")
                    save_image_dict(mult_img_dict, mult_img_dict_path)
                # if end
                data_start = time.time()
            # per train_loader end
            average_epoch_loss = epoch_loss / len(train_loader.dataset)
            epochs_losses.append(average_epoch_loss)
            os.makedirs(self.figure_dir, exist_ok=True)
            loss_plot(epochs_losses, os.path.join(self.figure_dir, "loss.png"), x_label="Epoch", y_label="Loss")
            loss_table(epochs_losses, os.path.join(self.figure_dir, "loss.xlsx"), y1_label="Epoch", y2_label="Loss")
            # per 1 epoch 计算PSNR
            if epoch % 5 == 0:
                self.model.eval()
                with torch.no_grad():
                    avg_psnr, avg_ssim = self.val_sample(val_loader, self.step)
                    psnr.append(avg_psnr)
                    ssim.append(avg_ssim)
                    loss_plot(psnr, os.path.join(self.figure_dir, "psnr.png"), y_label="PSNR")
                    loss_table(psnr, os.path.join(self.figure_dir, "psnr.xlsx"), y1_label="Epoch", y2_label="PSNR")
                    loss_plot(ssim, os.path.join(self.figure_dir, "ssim.png"), y_label="SSIM")
                    loss_table(ssim, os.path.join(self.figure_dir, "ssim.xlsx"), y1_label="Epoch", y2_label="SSIM")
                    self.logger_val.info(
                        "<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e} ssim: {:.4e}".format(self.start_epoch, self.step,
                                                                                      avg_psnr, avg_ssim)
                    )
                    if avg_psnr > best_psnr:
                        best_psnr = avg_psnr
                        ckpt_save_path = os.path.join(self.checkpoint_dir, self.args.name + "_" + "best")
                        self.logger.info("Saving best_psnr models and training states in {}.".format(ckpt_save_path))
                        save_checkpoint(
                            {
                                "loss": epochs_losses,
                                "psnr": psnr,
                                "ssim": ssim,
                                "epoch": self.start_epoch,
                                "step": self.step,
                                "state_dict": self.model.state_dict(),
                                "optimizer": self.optimizer.state_dict(),
                                "ema_helper": self.ema_helper.state_dict(),
                                "params": self.args,
                                "config": self.config,
                            },
                            filename=ckpt_save_path,
                        )
            # per 100 epoch save model
            if epoch % 100 == 0:
                ckpt_save_path = os.path.join(self.checkpoint_dir, self.args.name + "_epoch_" + str(self.start_epoch))
                self.logger.info("Saving models and training states in {}.".format(ckpt_save_path))
                save_checkpoint(
                    {
                        "loss": epochs_losses,
                        "psnr": psnr,
                        "ssim": ssim,
                        "epoch": self.start_epoch,
                        "step": self.step,
                        "state_dict": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "ema_helper": self.ema_helper.state_dict(),
                        "params": self.args,
                        "config": self.config,
                    },
                    filename=ckpt_save_path,
                )
            epoch_time = time.time() - epoch_start_time
            self.logger.info(
                f"currenr epoch: {self.start_epoch}/{self.config.training.n_epochs}\tloss: {average_epoch_loss:.8f}\tper epoch time: {epoch_time}"
            )
        self.start_epoch += 1

    ###### 第一种采样
    # q(xt|x0)
    def sample_forward(self, x0, t, eps):
        coef1 = self.extract(self.sqrt_alphas_bar, t)
        coef2 = self.extract(self.sqrt_one_minus_alphas_bar, t)
        # if eps is None:
        #     eps = torch.randn_like(x0)
        res = coef1 * x0 + coef2 * eps
        res = torch.clamp(res, -1, 1)
        return res

    # q(x0|xt)
    def sample_backward(
            self,
            x_t,
            x_cond,
            simple_var=True,
            clip_x0=True,
    ):
        with torch.no_grad():
            for t in reversed(range(self.num_timesteps)):
                x_t = self.sample_backward_step(x_t, x_cond, t, simple_var, clip_x0)
        return x_t

    def sample_backward_step(
            self,
            x_t,
            x_cond,
            t,
            simple_var=True,
            clip_x0=True,
    ):
        n = x_t.shape[0]
        # t_tensor = (torch.ones(n) * t).to(x_t.device)
        t_tensor = torch.tensor([t] * n, dtype=torch.long).to(x_t.device).unsqueeze(1)
        input = torch.cat([x_cond, x_t], dim=1)
        eps = self.model(input, t_tensor)
        if t == 0:
            noise = 0
        else:
            if simple_var:
                var = self.betas[t]
            else:
                var = (
                        (1 - self.alpha_bar[t - 1])
                        / (1 - self.alpha_bar[t])
                        * self.betas[t]
                )
            noise = torch.randn_like(x_t)
            noise *= torch.sqrt(var)

        if clip_x0:
            x_0 = (x_t - torch.sqrt(1 - self.alpha_bar[t]) * eps) / torch.sqrt(
                self.alpha_bar[t]
            )
            x_0 = torch.clip(x_0, -1, 1)
            mean = self.coef1[t] * x_t + self.coef2[t] * x_0
        else:
            mean = (
                           x_t - (1 - self.alphas[t]) / torch.sqrt(1 - self.alpha_bar[t]) * eps
                   ) / torch.sqrt(self.alphas[t])
        x_t = mean + noise

        return x_t

    def compute_alpha(self, beta, t):
        beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a

    ###### 第二种采样
    def generalized_steps(self, xt, x_cond, seq, b, eta=0.8):
        with torch.no_grad():
            n = xt.size(0)
            seq_next = [-1] + list(seq[:-1])
            x0_preds = []
            xs = [xt]
            for i, j in zip(reversed(seq), reversed(seq_next)):
                t = (torch.ones(n) * i).to(xt.device)
                # print(t.shape)
                next_t = (torch.ones(n) * j).to(xt.device)
                at = self.compute_alpha(b, t.long())
                at_next = self.compute_alpha(b, next_t.long())
                xt = xs[-1].to("cuda")

                et = self.model(torch.cat([x_cond, xt], dim=1), t.float())
                x_0 = (xt - et * (1 - at).sqrt()) / at.sqrt()

                x_0 = torch.clip(x_0, -1, 1)

                x0_preds.append(x_0.to("cpu"))
                c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                c2 = ((1 - at_next) - c1 ** 2).sqrt()
                xt_next = at_next.sqrt() * x_0 + c1 * torch.randn_like(x_0) + c2 * et
                xs.append(xt_next.to("cpu"))
        return xs, x0_preds

    def ddpm_steps(self, x, x_cond, seq, model, b, **kwargs):
        with torch.no_grad():
            n = x.size(0)
            seq_next = [-1] + list(seq[:-1])
            xs = [x]
            x0_preds = []
            betas = b
            for i, j in zip(reversed(seq), reversed(seq_next)):
                t = (torch.ones(n) * i).to(x.device)
                next_t = (torch.ones(n) * j).to(x.device)
                at = self.compute_alpha(betas, t.long())
                atm1 = self.compute_alpha(betas, next_t.long())
                beta_t = 1 - at / atm1
                x = xs[-1].to("cuda")

                output = model(torch.cat([x_cond, x], dim=1), t.float())
                e = output

                x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
                x0_from_e = torch.clamp(x0_from_e, -1, 1)
                x0_preds.append(x0_from_e.to("cpu"))
                mean_eps = (
                                   (atm1.sqrt() * beta_t) * x0_from_e
                                   + ((1 - beta_t).sqrt() * (1 - atm1)) * x
                           ) / (1.0 - at)

                mean = mean_eps
                noise = torch.randn_like(x)
                mask = 1 - (t == 0).float()
                mask = mask.view(-1, 1, 1, 1)
                logvar = beta_t.log()
                sample = mean + mask * torch.exp(0.5 * logvar) * noise
                xs.append(sample.to("cpu"))
        return xs, x0_preds

    # 采样一张图片
    def sample_image(
            self,
            x_cond,
            xt,
            last=True,
            sample_type="generalized",
            skip_type="uniform",
    ):
        """
        sample_type:采样类型 generalized 和 ddpm_noisy
        skip_type:步长跳过类型 uniform 和 quad
        """
        if sample_type == "generalized":
            if skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif skip_type == "quad":
                seq = (
                        np.linspace(
                            0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                        )
                        ** 2
                )
                seq = [int(s) for s in list(seq)]

            xt = self.generalized_steps(xt, x_cond, seq, self.betas, eta=0.01)
        elif sample_type == "ddpm_noisy":
            if skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif skip_type == "quad":
                seq = (
                        np.linspace(
                            0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                        )
                        ** 2
                )
                seq = [int(s) for s in list(seq)]

            xt = self.ddpm_steps(xt, seq, self.betas)

        if last:
            xt = xt[0][-1]
        return xt

    def visualize_forward(self, val_loader):
        import cv2
        import einops
        import numpy as np

        x, _ = next(iter(val_loader))[0]
        x = x.to(self.device)

        xts = []
        percents = torch.linspace(0, 0.99, 10)
        for percent in percents:
            t = torch.tensor([int(self.num_timesteps * percent)])
            t = t.unsqueeze(1)
            x_t = self.sample_forward(x[0, 6:, :, :], t)
            xts.append(x_t)
        res = torch.stack(xts, 0)
        res = einops.rearrange(res, "n1 n2 c h w -> (n2 h) (n1 w) c")
        res = (res.clip(-1, 1) + 1) / 2 * 255
        res = res.cpu().numpy().astype(np.uint8)

        cv2.imwrite(
            "/data2/wait/bisheCode/DDPM_Fusion/images/diffusion_forward.jpg", res
        )

    # 训练的时候进行验证集采样来看psnr ssim 以及采样结果
    def val_sample(self, val_loader, epoch):
        self.model.eval()
        image_floder = image_folder = os.path.join(
            self.val_sample_dir, "epoch_{:04d}".format(epoch)
        )
        os.makedirs(image_floder, exist_ok=True)

        with torch.no_grad():
            self.logger.info(f"Processing val images at epoch: {epoch}")
            per_img_psnr = 0.0
            per_img_ssim = 0.0
            for i, (x, y) in enumerate(tqdm(val_loader)):
                # print(f"flat 前 x.shape: {x.shape}")
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                # print(f"flat 后 x.shape: {x.shape}")

                n = x.shape[0]
                x_cond = x[:, :6, :, :].to(self.device)
                x_gt = x[:, 6:, ::]
                x_cond = data_transform(x_cond)
                shape = x_gt.shape
                x = torch.randn(shape).to(self.device)
                # 第一个办法
                pred_x = self.sample_image(x_cond, x)
                # 第二个办法
                # pred_x = self.sample_backward(
                #     x_t=x, x_cond=x_cond, simple_var=True, clip_x0=True
                # )
                pred_x = torch.clip(pred_x, -1, 1)

                pred_x = inverse_data_transform(pred_x)
                x_cond = inverse_data_transform(x_cond)

                for i in range(n):
                    per_img_psnr += metrics.calculate_psnr(
                        pred_x[i].permute(1, 2, 0).numpy() * 255.0,
                        x_gt[i].permute(1, 2, 0).numpy() * 255.0,
                        test_y_channel=True,
                    )
                    per_img_ssim += metrics.calculate_ssim(
                        pred_x[i].permute(1, 2, 0).numpy() * 255.0,
                        x_gt[i].permute(1, 2, 0).numpy() * 255.0,
                    )

                    save_image(
                        x_cond[:, :3, :, :][i],
                        os.path.join(
                            image_folder, "{}_degraded1.png".format(y[i][:-4])
                        ),
                    )
                    save_image(
                        x_cond[:, 3:, :, :][i],
                        os.path.join(
                            image_folder, "{}_degraded2.png".format(y[i][:-4])
                        ),
                    )
                    save_image(
                        pred_x[i],
                        os.path.join(image_folder, "{}_fake.png".format(y[i][:-4])),
                    )
                    save_image(
                        x_gt[i],
                        os.path.join(image_folder, "{}_gt.png".format(y[i][:-4])),
                    )
        avg_psnr = per_img_psnr / len(val_loader.dataset)
        avg_ssim = per_img_ssim / len(val_loader.dataset)
        return avg_psnr, avg_ssim

    # 对测试集进行采样
    def test_sample(self, test_loader, type):
        """
        type: 路径名
        """
        self.load_ddm_ckpt(self.args.resume, ema=False, train=False)
        self.model.eval()
        image_folder = self.test_sample_dir
        with torch.no_grad():
            self.logger.info(f"Processing test images at step: {self.start_epoch}")
        test_loader1 = tqdm(test_loader)
        for _, (x, y) in enumerate(test_loader1):
            x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
            n = x.shape[0]
            x_cond = x[:, :6, :, :].to(self.device)
            x_cond = data_transform(x_cond)
            shape = x_cond[:, :3, :, :].shape
            xt = torch.randn(shape, device=self.device)
            # 第一个办法
            pred_x = self.sample_image(
                x_cond,
                xt,
                last=True,
                sample_type="generalized",
                skip_type="uniform",
            )
            # 第二个办法
            # pred_x = self.sample_backward(
            #     x_t=x, x_cond=x_cond, simple_var=True, clip_x0=True
            # )
            pred_x = torch.clip(pred_x, -1, 1)
            pred_x = inverse_data_transform(pred_x)
            x_cond = inverse_data_transform(x_cond)

            for i in range(n):
                ml_img = [
                    x_cond[i, :3, :, :].detach().float().cpu(),
                    x_cond[i, 3:, :, :].detach().float().cpu(),
                    pred_x[i].detach().float().cpu(),
                ]
                ml_img = make_grid(ml_img, nrow=3, padding=2)
                os.makedirs(os.path.join(image_folder, type, "grid"), exist_ok=True)
                save_img(
                    ml_img,
                    os.path.join(image_folder, type, "grid", y[i]),
                    nrow=3,
                    padding=2,
                )
                os.makedirs(os.path.join(image_folder, type, "pred"), exist_ok=True)
                save_image(pred_x[i], os.path.join(image_folder, type, "pred", y[i]))
                os.makedirs(os.path.join(image_folder, type, "cond1"), exist_ok=True)
                save_image(
                    x_cond[:, :3, :, :][i],
                    os.path.join(image_folder, type, "cond1", y[i]),
                )
                os.makedirs(os.path.join(image_folder, type, "cond2"), exist_ok=True)
                save_image(
                    x_cond[:, 3:, :, :][i],
                    os.path.join(image_folder, type, "cond2", y[i]),
                )
                test_loader1.set_description("{} | {}".format(self.args.name, y[i]))

    # 采样融合结果
    def Fusion_sample(self, fusion_loader, type):
        """
        type: 路径名
        """
        self.load_ddm_ckpt(self.args.resume, ema=False, train=False)
        image_folder = os.path.join(self.fusion_dir, type)
        os.makedirs(image_folder, exist_ok=True)

        self.model.eval()
        with torch.no_grad():
            self.logger.info(f"Processing test images at step: {self.start_epoch}")
        loader = tqdm(fusion_loader)
        for _, (x, y) in enumerate(loader):
            x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
            n = x.shape[0]
            x_cond = x[:, :6, :, :].to(self.device)
            x_cond = data_transform(x_cond)
            shape = x_cond[:, :3, :, :].shape
            xt = torch.randn(shape, device=self.device)
            # 第一个办法
            pred_x = self.sample_image(
                x_cond,
                xt,
                last=True,
                sample_type="generalized",
                skip_type="uniform",
            )
            # 第二个办法
            # pred_x = self.sample_backward(
            #     x_t=x, x_cond=x_cond, simple_var=True, clip_x0=True
            # )
            pred_x = torch.clip(pred_x, -1, 1)
            pred_x = inverse_data_transform(pred_x)
            x_cond = inverse_data_transform(x_cond)

            for i in range(n):
                ml_img = [
                    x_cond[i, :3, :, :].detach().float().cpu(),
                    x_cond[i, 3:, :, :].detach().float().cpu(),
                    pred_x[i].detach().float().cpu(),
                ]
                ml_img = make_grid(ml_img, nrow=3, padding=2)
                os.makedirs(os.path.join(image_folder, "grid"), exist_ok=True)
                save_img(
                    ml_img,
                    os.path.join(image_folder, "grid", y[i]),
                    nrow=3,
                    padding=2,
                )
                os.makedirs(os.path.join(image_folder, "pred"), exist_ok=True)
                save_image(pred_x[i], os.path.join(image_folder, type, "pred", y[i]))
                os.makedirs(os.path.join(image_folder, "ir"), exist_ok=True)
                save_image(
                    x_cond[:, :3, :, :][i],
                    os.path.join(image_folder, "ir", y[i]),
                )
                os.makedirs(os.path.join(image_folder, "vi"), exist_ok=True)
                save_image(
                    x_cond[:, 3:, :, :][i],
                    os.path.join(image_folder, "vi", y[i]),
                )
                loader.set_description("{} | {}".format(self.args.name, y[i]))


if __name__ == '__main__':
    path = os.path.join(os.getcwd(), "test")
    print(path)
