from PIL import Image
import os

# 指定输入和输出文件夹路径
input_folder = '/data2/wait/bisheCode/DDPM_Fusion/example'
output_folder = '/data2/wait/bisheCode/DDPM_Fusion'

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 遍历输入文件夹中的图像文件
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # 打开图像文件
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path)

        # 裁剪图像
        width, height = img.size
        # left = (width - 640) / 2
        # top = (height - 420) / 2
        # right = (width + 640) / 2
        # bottom = (height + 420) / 2
        # 计算裁剪区域
        left = (width - 256) / 2
        top = (height - 256) / 2
        right = (width + 256) / 2
        bottom = (height + 256) / 2
        cropped_img = img.crop((left, top, right, bottom))


        # 保存裁剪后的图像到输出文件夹
        output_path = os.path.join(output_folder, filename)
        cropped_img.save(output_path)

        # 关闭图像文件
        img.close()

print("裁剪完成")