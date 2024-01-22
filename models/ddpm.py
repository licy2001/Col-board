# -*- coding:utf-8 -*-
import os, sys

sys.path.append("/data2/wait/bisheCode/DDPM_Fusion")
import logging
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
from models.unet import UNet
from models.ema import EMAHelper, EMA
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
from functions.metrics import calculate_psnr, calculate_ssim


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
        if self.config.model.ema:
            self.ema = EMA(self.model, decay=self.config.model.ema_rate)
            self.ema.register()
            # self.ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            # self.ema_helper.register(self.model)
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
        betas = betas.to(torch.float32).to(self.device)
        self.betas = betas
        self.num_timesteps = betas.shape[0]
        self.alphas = (1.0 - self.betas)
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)
        self.alphas_bar_prev = F.pad(self.alphas_bar[:-1], (1, 0), value=1.0)
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1.0 - self.alphas_bar)
        # self.log_one_minus_alphas_bar = torch.log(1.0 - self.alphas_bar)
        self.sqrt_recip_alphas_bar = torch.sqrt(1.0 / self.alphas_bar)
        # self.sqrt_recipm1_alphas_bar = torch.sqrt(1.0 / self.alphas_bar - 1)
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.p_variance = (
                self.betas * (1.0 - self.alphas_bar_prev) / (1.0 - self.alphas_bar)
        )
        # self.p_log_variance_clipped = torch.log(self.p_variance.clamp(min=1e-20))

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
        if ema:
            # self.ema_helper.ema(self.model)
            self.ema.load_state_dict(checkpoint["ema"])
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
        self.fusion_dir = os.path.join(self.root_dir, config.path.fusion)

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

    def get_model_input(self, xt, x_cond):
        # print(self.args.concat_type)
        if self.args.concat_type == "ABX":
            model_input = torch.cat([x_cond, xt], dim=1)
        elif self.args.concat_type == "AXB":
            model_input = torch.cat([x_cond[:, :3, :, :], xt, x_cond[:, 3:, :, :]], dim=1)
        else:
            model_input = torch.cat([xt, x_cond], dim=1)
        return model_input

    def get_loss(self, x0, x_cond):
        """
        计算MSE损失
        """
        t = self.sample_timestep(x0.shape[0], symmetric=False, big_range=True)
        eps = torch.randn_like(x0).to(self.device)
        eps = torch.clamp(eps, -1, 1)
        xt = self.sample_forward(x0, t, eps)
        model_input = self.get_model_input(x_cond, xt)
        eps_theta = self.model(model_input, t.float())
        eps_theta = torch.clamp(eps_theta, -1, 1)

        x0_coef1 = self.extract(self.sqrt_recip_alphas_bar, t)
        x0_coef2 = self.extract(self.sqrt_one_minus_alphas_bar, t)
        pred_x0 = x0_coef1 * (xt - x0_coef2 * eps_theta)
        pred_x0 = torch.clamp(pred_x0, -1, 1)
        loss_fn = torch.nn.MSELoss(reduction="mean")
        # loss_fn = torch.nn.MSELoss(size_average=True, reduce=True, reduction="sum")
        loss = loss_fn(eps_theta, eps) * self.config.training.batch_size
        return loss, xt, eps, eps_theta, pred_x0

    def train(self, DATASET):
        # cudnn.benchmark = True
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
            # 开启梯度更新
            self.model.train()
            for i, (x, y) in enumerate(tqdm(train_loader)):
                self.optimizer.zero_grad()
                self.step += 1
                # n = current_batch_size
                data_time += time.time() - data_start

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
                loss.backward()
                try:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.optim.grad_clip
                    )
                except Exception:
                    pass
                self.optimizer.step()

                if self.config.model.ema:
                    self.ema.update()
                    # self.ema_helper.update(self.model)

                # epoch_loss += loss.item() * current_batch_size
                epoch_loss += loss.item()
                if self.step % 100 == 0 or self.step == 1:
                    mult_img_dict = {
                        'cond1': inverse_data_transform(x[0, :3, :, :].detach()),
                        'cond2': inverse_data_transform(x[0, 3:6, :, :].detach()),
                        'xt': inverse_data_transform(x_t[0, ::].detach()),
                        'noise': inverse_data_transform(eps[0, ::].detach()),
                        'pred_noise': inverse_data_transform(pred_noise[0, ::].detach()),
                        'pred_x0': inverse_data_transform(pred_x0[0, ::].detach()),
                        'x0': inverse_data_transform(x[0, 6:, ::].detach())
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

            ###### per 1 epoch 计算PSNR
            if epoch % 1 == 0:
                print(f"start val sample at epoch: {epoch}")
                avg_psnr, avg_ssim = self.val_sample(val_loader, epoch)
                psnr.append(avg_psnr)
                ssim.append(avg_ssim)
                loss_plot(psnr, os.path.join(self.figure_dir, "psnr.png"), y_label="PSNR")
                loss_table(psnr, os.path.join(self.figure_dir, "psnr.xlsx"), y1_label="Epoch", y2_label="PSNR")
                loss_plot(ssim, os.path.join(self.figure_dir, "ssim.png"), y_label="SSIM")
                loss_table(ssim, os.path.join(self.figure_dir, "ssim.xlsx"), y1_label="Epoch", y2_label="SSIM")
                self.logger_val.info(
                    "<epoch:{}\titer:{}>\tpsnr: {:.4f}\tssim: {:.4f}".format(epoch, self.step,
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
                            "epoch": epoch,
                            "step": self.step,
                            "state_dict": self.model.state_dict(),  # 模型的状态字典
                            "optimizer": self.optimizer.state_dict(),
                            "ema": self.ema.state_dict(),  # ema的状态字典
                            "params": self.args,
                            "config": self.config,
                        },
                        filename=ckpt_save_path,
                    )
            # per 100 epoch save model
            if epoch % 100 == 0:
                ckpt_save_path = os.path.join(self.checkpoint_dir, self.args.name + "_epoch_" + str(epoch))
                self.logger.info("Saving models and training states in {}.".format(ckpt_save_path))
                save_checkpoint(
                    {
                        "loss": epochs_losses,
                        "psnr": psnr,
                        "ssim": ssim,
                        "epoch": epoch,
                        "step": self.step,
                        "state_dict": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "ema": self.ema.state_dict(),
                        "params": self.args,
                        "config": self.config,
                    },
                    filename=ckpt_save_path,
                )
            epoch_time = time.time() - epoch_start_time
            self.logger.info(
                f"epoch: {epoch}/{self.config.training.n_epochs}\tloss: {average_epoch_loss:.8f}\tper epoch time: {epoch_time}"
            )

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
    @torch.no_grad()
    def sample_backward(
            self,
            xt,
            x_cond,
            simple_var=True,
            clip_x0=True,
    ):
        n = xt.shape[0]

        for i in reversed(tqdm(range(self.num_timesteps))):
            t = (torch.ones(n) * i).to(xt.device)
            xt = self.sample_backward_step(xt, x_cond, t, i, simple_var, clip_x0)
        return xt

    @torch.no_grad()
    def sample_backward_step(
            self,
            xt,
            x_cond,
            t,
            i,
            simple_var=True,
            clip_x0=True,
    ):
        model_input = self.get_model_input(x_cond, xt)
        eps_theta = self.model(model_input, t)
        if i == 0:
            noise = 0
        else:
            if simple_var:
                var = self.extract(self.betas, t)
            else:
                var = self.extract(self.p_variance, t)
            noise = torch.randn_like(xt)
            noise = torch.clamp(noise, -1, 1)
            noise *= torch.sqrt(var)

        if clip_x0:
            x0_coef1 = self.extract(self.sqrt_recip_alphas_bar, t)
            x0_coef2 = self.extract(self.sqrt_one_minus_alphas_bar, t)
            x_0 = x0_coef1 * (xt - x0_coef2 * eps_theta)
            x_0 = torch.clip(x_0, -1, 1)
            mean = self.extract(self.p_mean_coef1, t) * xt + self.extract(self.p_mean_coef2, t) * x_0
        else:
            mu_coef1 = 1 / self.extract(self.sqrt_alphas, t)
            mu_coef2 = (1 - self.extract(self.alphas, t)) / self.extract(self.sqrt_one_minus_alphas_bar, t)
            mean = mu_coef1 * (xt - mu_coef2 * eps_theta)

        x_t = mean + noise
        return x_t

    ###### 第二种采样
    @torch.no_grad()
    def get_sample_step(self):
        """
        Calculate time step size, it skips some steps
        """
        skip = self.num_timesteps // self.args.timesteps
        time_step = torch.arange(0, self.num_timesteps, skip).long() + 1
        time_step = reversed(torch.cat((torch.tensor([0], dtype=torch.long), time_step)))
        time_step = list(zip(time_step[:-1], time_step[1:]))
        return time_step

    @torch.no_grad()
    def ddim_sample(self, xt, x_cond, eta=0):
        # self.logger_val.info(msg=f"DDIM Sampling images....")
        # self.model.eval()
        n = xt.shape[0]
        # x0_preds = []
        # xs = [xt]
        # get DDIM time_step
        time_step = self.get_sample_step()
        # The list of current time and previous time
        for i, p_i in tqdm(time_step):
            # Time step, creating a tensor of size n
            t = (torch.ones(n) * i).long().to(self.device)
            # Previous time step, creating a tensor of size n
            p_t = (torch.ones(n) * p_i).long().to(self.device)
            # Expand to a 4-dimensional tensor, and get the value according to the time step t
            alpha_t = self.alphas_bar[t][:, None, None, None].to(self.device)
            alpha_prev = self.alphas_bar[p_t][:, None, None, None].to(self.device)
            if i > 1:
                noise = torch.randn_like(xt, dtype=torch.float32)
            else:
                noise = torch.zeros_like(xt, dtype=torch.float32)
            noise = torch.clamp(noise, -1, 1) # 加的
            model_input = self.get_model_input(xt, x_cond)
            eps_theta = self.model(model_input, t.float())
            eps_theta = torch.clamp(eps_theta, -1, 1) # 加的
            # Calculation formula
            # xt = xs[-1].to(self.device)
            x0_t = (xt - (eps_theta * torch.sqrt(1 - alpha_t))) / torch.sqrt(alpha_t)
            x0_t = torch.clamp(x0_t, -1, 1)
            # x0_preds.append(x0_t.to('cpu'))
            c1 = eta * torch.sqrt((1 - alpha_t / alpha_prev) * (1 - alpha_prev) / (1 - alpha_t))
            c2 = torch.sqrt((1 - alpha_prev) - c1 ** 2)
            xt = torch.sqrt(alpha_prev) * x0_t + c2 * eps_theta + c1 * noise
            xt = torch.clamp(xt, -1, 1) # 加的
            # xs.append(xt.to('cpu'))
        return xt

    @torch.no_grad()
    def ddpm_steps(self, xt, x_cond, seq):
        with torch.no_grad():
            n = xt.size(0)
            seq_next = [-1] + list(seq[:-1])
            xs = [xt]
            x0_preds = []
            for i, j in zip(reversed(seq), reversed(seq_next)):
                t = (torch.ones(n) * i).to(xt.device)
                next_t = (torch.ones(n) * j).to(xt.device)
                at = self.extract(self.alphas_bar, t)
                atm1 = self.extract(self.alphas_bar, next_t)
                beta_t = 1 - at / atm1
                xt = xs[-1].to("cuda:1")
                model_input = self.get_model_input(x_cond, xt)
                et = self.model(model_input, t.float())

                x_0 = (xt - et * torch.sqrt(1 - at)) / torch.sqrt(at)
                x_0 = torch.clamp(x_0, -1, 1)
                x0_preds.append(x_0.to("cpu"))
                mean_eps = ((torch.sqrt(atm1) * beta_t) * x_0 + (torch.sqrt(1 - beta_t) * (1 - atm1)) * xt) / (1.0 - at)

                noise = torch.randn_like(xt)
                noise = torch.clamp(noise, -1, 1)
                mask = 1 - (t == 0).float()
                mask = mask.view(-1, 1, 1, 1)
                logvar = torch.log(beta_t)
                sample = mean_eps + mask * torch.exp(0.5 * logvar) * noise
                sample = torch.clamp(sample, -1, 1)
                xs.append(sample.to("cpu"))
        return xs, x0_preds

    # 采样一张图片
    @torch.no_grad()
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

            xt = self.generalized_steps(xt, x_cond, seq, eta=0.8)
        elif sample_type == "ddpm_noisy":
            if skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif skip_type == "quad":
                # else:
                seq = (
                        np.linspace(
                            0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                        )
                        ** 2
                )
                seq = [int(s) for s in list(seq)]
            xt = self.ddpm_steps(xt, x_cond, seq)

        if last:
            # xt = xt[0]
            xt = xt[0][-1]
        return xt

    def visualize_forward(self, val_loader):
        import cv2
        import einops
        import numpy as np

        x, _ = next(iter(val_loader))[0]
        x = x.to(self.device)
        x0 = x[0, 6:, :, :]
        n = x0.shape[0]
        xts = []
        percents = torch.linspace(0, 0.99, 10)

        for percent in percents:
            i = int(self.num_timesteps * percent)
            t = (torch.ones(n) * i).to(x.device)
            eps = torch.randn_like(x0)
            eps = torch.clamp(eps, -1, 1)
            xt = self.sample_forward(x0, t, eps)
            xts.append(xt)
        res = torch.stack(xts, 0)
        res = einops.rearrange(res, "n1 n2 c h w -> (n2 h) (n1 w) c")
        res = (res.clip(-1, 1) + 1) / 2 * 255
        res = res.cpu().numpy().astype(np.uint8)

        cv2.imwrite(
            "/data2/wait/bisheCode/DDPM_Fusion/diffusion_forward.jpg", res
        )

    # 训练的时候进行验证集采样来看psnr ssim 以及采样结果
    @torch.no_grad()
    def val_sample(self, val_loader, epoch):
        self.ema.apply_shadow()  # 应用 EMA 影子参数
        self.model.eval()
        image_floder = image_folder = os.path.join(self.val_sample_dir, "epoch_{:04d}".format(epoch))
        os.makedirs(image_floder, exist_ok=True)
        self.logger.info(f"Processing val images at epoch: {epoch}")
        per_img_psnr = 0.0
        per_img_ssim = 0.0
        for i, (x, y) in enumerate(tqdm(val_loader)):
            n = x.shape[0]
            x = data_transform(x)
            x = x.to(self.device)
            x_cond = x[:, :6, :, :]
            x_gt = x[:, 6:, ::]
            # x_cond = data_transform(x_cond)
            xt = torch.randn(x_gt.shape, dtype=torch.float32).to(self.device)
            pred_x = self.ddim_sample(xt, x_cond, eta=0)
            # 第一个办法
            # pred_x = self.sample_image(
            #     x_cond,
            #     xt,
            #     last=True,
            #     sample_type="generalized",
            #     skip_type="uniform",
            # )
            # 第二个办法
            # pred_x = self.sample_backward(xt, x_cond, simple_var=True, clip_x0=True)
            pred_x = inverse_data_transform(pred_x)
            x_cond = inverse_data_transform(x_cond)
            x_gt = inverse_data_transform(x_gt)
            for i in range(n):
                per_img_psnr += calculate_psnr(pred_x[i], x_gt[i], test_y_channel=True)
                per_img_ssim += calculate_ssim(pred_x[i], x_gt[i])
                mult_img_dict = {
                    'cond1': x_cond[i, :3, :, :].detach(),
                    'cond2': x_cond[i, 3:, :, :].detach(),
                    'gt': x_gt[i].detach(),
                    'pred': pred_x[i].detach(),
                }
                mult_img_dict_path = os.path.join(os.path.join(image_folder, y[i]))
                save_image_dict(mult_img_dict, mult_img_dict_path)

        avg_psnr = per_img_psnr / len(val_loader.dataset)
        avg_ssim = per_img_ssim / len(val_loader.dataset)
        self.ema.restore()  # 恢复原始模型参数
        return avg_psnr, avg_ssim

    # 对测试集进行采样
    @torch.no_grad()
    def test_sample(self, test_loader, type):
        """
        type: 路径名
        """
        self.load_ddm_ckpt(self.args.resume, ema=True, train=True)
        self.ema.apply_shadow()  # 应用 EMA 影子参数获得平滑后的参数
        self.model.eval()
        image_folder = self.test_sample_dir
        self.logger.info(f"Processing test images at step: {self.start_epoch}")
        per_img_psnr = 0.0
        per_img_ssim = 0.0
        for _, (x, y) in enumerate(tqdm(test_loader)):
            n = x.shape[0]
            x = data_transform(x)
            x_cond = x[:, :6, :, :].to(self.device)
            x_gt = x[:, 6:, ::].to(self.device)
            shape = x_gt.shape
            xt = torch.randn(shape, device=self.device)
            xt = torch.clamp(xt, -1, 1)
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
            pred_x = inverse_data_transform(pred_x)
            x_cond = inverse_data_transform(x_cond)
            x_gt = inverse_data_transform(x_gt)
            for i in range(n):
                per_img_psnr += calculate_psnr(pred_x[i], x_gt[i], test_y_channel=True)
                per_img_ssim += calculate_ssim(pred_x[i], x_gt[i])
                mult_img_dict = {
                    'cond1': x_cond[i, :3, :, :].detach(),
                    'cond2': x_cond[i, 3:, :, :].detach(),
                    'gt': x_gt[i].detach(),
                    'pred': pred_x[i].detach(),
                }
                os.makedirs(os.path.join(image_folder, type, "grid"), exist_ok=True)
                mult_img_dict_path = os.path.join(os.path.join(image_folder, type, "grid", y[i]))
                save_image_dict(mult_img_dict, mult_img_dict_path)

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
        avg_psnr = per_img_psnr / len(test_loader.dataset)
        avg_ssim = per_img_ssim / len(test_loader.dataset)
        self.ema.restore()  # 恢复原始模型参数
        return avg_psnr, avg_ssim

    # 采样融合结果
    @torch.no_grad()
    def Fusion_sample(self, fusion_loader, type):
        """
        type: 路径名
        """
        self.load_ddm_ckpt(self.args.resume, ema=True, train=True)
        self.ema.apply_shadow()  # 应用 EMA 影子参数获得平滑后的参数
        self.model.eval()
        image_folder = os.path.join(self.fusion_dir, type)
        os.makedirs(image_folder, exist_ok=True)
        self.logger_val.info(f"Processing test images at step: {self.start_epoch}")
        for _, (x, y) in enumerate(tqdm(fusion_loader)):
            n = x.shape[0]
            x = data_transform(x)
            x_cond = x.to(self.device)
            shape = x_cond[:, :3, :, :].shape
            xt = torch.randn(shape, device=self.device)
            xt = torch.clamp(xt, -1, 1)
            # xt = xt.to(self.device)
            # 第一个办法
            # pred_x = self.sample_image(
            #     x_cond,
            #     xt,
            #     last=True,
            #     sample_type="generalized",
            #     skip_type="uniform",
            # )
            # 第二个办法
            pred_x = self.sample_backward(xt, x_cond, simple_var=True, clip_x0=True)
            pred_x = inverse_data_transform(pred_x)
            x_cond = inverse_data_transform(x_cond)

            for i in range(n):
                mult_img_dict = {
                    'ir': x_cond[i, :3, :, :].detach(),
                    'vi': x_cond[i, 3:, :, :].detach(),
                    'fusion': pred_x[i].detach(),
                }
                os.makedirs(os.path.join(image_folder, "grid"), exist_ok=True)
                mult_img_dict_path = os.path.join(os.path.join(image_folder, "grid", y[i]))
                save_image_dict(mult_img_dict, mult_img_dict_path)
                os.makedirs(os.path.join(image_folder, "pred"), exist_ok=True)
                save_image(pred_x[i], os.path.join(image_folder, "pred", y[i]))
                os.makedirs(os.path.join(image_folder, "ir"), exist_ok=True)
                save_image(x_cond[:, :3, :, :][i], os.path.join(image_folder, "ir", y[i]))
                os.makedirs(os.path.join(image_folder, "vi"), exist_ok=True)
                save_image(x_cond[:, 3:, :, :][i], os.path.join(image_folder, "vi", y[i]))
        self.ema.restore()  # 恢复原始模型参数


if __name__ == '__main__':
    path = os.path.join(os.getcwd(), "test")
    print(path)
