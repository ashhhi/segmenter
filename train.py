from torch import optim
from create_dataset import *
from torch.utils.data import DataLoader
from model.factory import create_segmenter
from config import load_config
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
from tqdm import tqdm
from model import gpu_wrapper as ptu
import torch
import os

is_eval = True

transform = A.Compose([
    A.Resize(224, 224),  # 调整图像大小为
    A.HorizontalFlip(p=0.5),  # 水平翻转概率为 0.5
    A.RandomBrightnessContrast(p=0.2),  # 随机亮度和对比度
    A.Normalize(),  # 归一化
    ToTensorV2()  # 将图像转换为 PyTorch 张量
])


save_path = 'checkpoint'
dataset_path = '/Users/shijunshen/Documents/Code/dataset/Smart-Farm-All/Final-Dataset'

dataset = SmartFarm(dataset_path, transform=transform)

train_data = DataLoader(dataset, batch_size=4, shuffle=True)

# 定义模型
backbone = "vit_base_patch8_384"
cfg = load_config()
model_cfg = cfg["model"][backbone]
model_cfg["image_size"] = (224,224)
decoder_cfg = cfg["decoder"]['mask_transformer']
decoder_cfg["name"] = 'mask_transformer'
model_cfg['decoder'] = decoder_cfg
model_cfg['n_cls'] = 3
model_cfg["backbone"] = backbone

model = create_segmenter(model_cfg)
model = model.to(ptu.device)



if is_eval:
    model.eval()
    state_dict = torch.load('checkpoint/model_25.pth', map_location=torch.device(ptu.device))
    model.load_state_dict(state_dict)

    for d in train_data:
        sample, target = d
        sample = sample.to(ptu.device)
        target = target.to(ptu.device)
        pred = model(sample).argmax(dim=1)
        break

    import matplotlib.pyplot as plt
    import numpy as np

    # 定义颜色映射
    colors = [
        [0, 0, 255],  # 类别 0 对应蓝色
        [0, 255, 0],  # 类别 1 对应绿色
        [255, 0, 0]  # 类别 2 对应红色
    ]


    # 将预测和目标转换为可视化图像
    def visualize_images(image_tensor):
        images = []
        for i in range(image_tensor.size(0)):
            image = torch.zeros(3, image_tensor.size(1), image_tensor.size(2))
            for j in range(len(colors)):
                image[0][image_tensor[i] == j] = colors[j][0] / 255.0
                image[1][image_tensor[i] == j] = colors[j][1] / 255.0
                image[2][image_tensor[i] == j] = colors[j][2] / 255.0
            images.append(image)

        return images


    # 可视化预测图像
    pred_images = visualize_images(pred)
    for i, image in enumerate(pred_images):
        plt.subplot(1, 4, i + 1)
        plt.imshow(image.permute(1, 2, 0).numpy())
        plt.axis('off')
        plt.title(f'Pred Image {i + 1}')

    plt.show()

    # 可视化目标图像
    target_images = visualize_images(target)
    for i, image in enumerate(target_images):
        plt.subplot(1, 4, i + 1)
        plt.imshow(image.permute(1, 2, 0).numpy())
        plt.axis('off')
        plt.title(f'Target Image {i + 1}')

    plt.show()

else:
    # 定义loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
    epoch = 100

    for e in range(epoch):
        t = tqdm(train_data)
        for d in t:
            optimizer.zero_grad()
            sample, target = d
            sample = sample.to(cpu)
            target = target.to(ptu.device)

            target = target.long()
            pred = model(sample)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            t.set_postfix({'loss': loss.item()})

        if e % 10 == 0:
            torch.save(model.state_dict(), os.path.join(save_path, f'model_{e}.pth'))
