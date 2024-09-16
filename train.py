from torch import optim
from create_dataset import *
from torch.utils.data import DataLoader
from model.factory import create_segmenter
from config import load_config
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
from tqdm import tqdm

transform = A.Compose([
    A.Resize(224, 224),  # 调整图像大小为
    A.HorizontalFlip(p=0.5),  # 水平翻转概率为 0.5
    A.RandomBrightnessContrast(p=0.2),  # 随机亮度和对比度
    A.Normalize(),  # 归一化
    ToTensorV2()  # 将图像转换为 PyTorch 张量
])



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

# 定义loss
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
epoch = 100

t = tqdm(train_data)
for e in range(epoch):
    for d in t:
        optimizer.zero_grad()
        sample, target = d
        target = target.long()
        pred = model(sample)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        t.set_postfix({'loss': loss.item()})
