import torch
import torchvision.transforms as transforms
from pytorch_msssim import ssim

def calculate_psnr(img1, img2, test_y_channel=False):
    """
    计算两个图像张量的psnr
    """
    # 确保图像张量在同一设备上，并且有相同的数据类型
    img1 = img1.to(torch.float32).cpu()
    img2 = img2.to(torch.float32).cpu()
    if test_y_channel:
        psnr = calculate_psnr_y_channel(img1, img2)
        return psnr
    else:
        mse = torch.mean((img1 - img2) ** 2)
        if mse == 0:
            return 100
        max_pixel = 1  # 255.0
        psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
        return psnr.item()


def calculate_psnr_y_channel(img1, img2):
    # 创建一个 ToPILImage 对象
    to_pil_img = transforms.ToPILImage()

    # 将张量转换为 PIL 图像
    img1 = to_pil_img(img1)
    img2 = to_pil_img(img2)

    # 将图像转换为 YCbCr 色彩空间
    img1 = img1.convert('YCbCr')
    img2 = img2.convert('YCbCr')

    # 分离 Y 通道
    y1, _, _ = img1.split()
    y2, _, _ = img2.split()

    # 将 PIL 图像转换为张量
    to_tensor = transforms.ToTensor()
    y1 = to_tensor(y1)
    y2 = to_tensor(y2)

    # 计算 PSNR
    mse = torch.mean((y1 - y2) ** 2)
    if mse == 0:
        return 100
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()

# img1 = torch.rand(3, 256, 256)
# img2 = torch.rand(3, 256, 256)
# psnr = calculate_psnr_y_channel(img1, img2)
# print(psnr)



def calculate_ssim(img1, img2):
    # 确保图像张量在同一设备上，并且有相同的数据类型
    img1 = img1.unsqueeze(0).to(torch.float32).cpu()
    img2 = img2.unsqueeze(0).to(torch.float32).cpu()

    # 计算 SSIM
    ssim_value = ssim(img1, img2, data_range=1.0, size_average=True)  # data_range取决于图像的范围，如果是[0,1]则为1.0，如果是[0,255]则为255.0
    return ssim_value.item()


if __name__ == '__main__':
    import PIL.Image as Image
    from torchvision.transforms import ToTensor
    img1 = Image.open("/data2/wait/bisheCode/DDPM_Fusion/results/CoCo/Fusion/LLVIP_coco_clip_net_x0_xt_optim/pred/010001.jpg")
    img2 = Image.open("/data2/wait/bisheCode/DDPM_Fusion/results/CoCo/Fusion/LLVIP_coco_clip_net_x0_optim/pred/010001.jpg")

    img1 = ToTensor()(img1)
    img2 = ToTensor()(img2)

    psnr = calculate_psnr(img1, img2, test_y_channel=True)
    print(f"psnr: {psnr}")
