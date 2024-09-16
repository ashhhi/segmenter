from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np

class SmartFarm(Dataset):
    def __init__(self, root_path, transform):
        self.img_path = os.path.join(root_path, 'img')
        self.mask_path = os.path.join(root_path, 'mask')
        self.transform = transform
        self.img_name = []
        self.mask_name = []
        for _, _, filename in os.walk(self.img_path):
            for name in filename:
                if name.endswith('.jpg'):
                    self.img_name.append(name)
                    self.mask_name.append(name.replace('.jpg', '.png'))

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.img_path, self.img_name[idx]))
        mask = Image.open(os.path.join(self.mask_path, self.mask_name[idx]))
        img = np.array(img)
        mask = np.array(mask)
        if self.transform:
            transformed = self.transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']


        return img, mask


