import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import albumentations
import cv2
from config import config

class HappyWhaleDataset(Dataset):

    def __init__(self, pd, trainFlag):
        self.pd = pd
        self.trainFlag = trainFlag
        if self.trainFlag:
            self.transform = albumentations.Compose([
                albumentations.Resize(config['img_size'], config['img_size']),
                albumentations.VerticalFlip(),
                albumentations.HorizontalFlip(),
                albumentations.Rotate(),
                albumentations.RandomBrightnessContrast(),
                albumentations.Normalize()
            ])
        else:
            self.transform = albumentations.Compose([
                albumentations.Normalize()
            ])

    def __len__(self):
        return self.pd.shape[0]

    def __getitem__(self, index):
        img_name = self.pd.iloc[index]['image']
        img_dir = self.pd.iloc[index]['path']

        img = cv2.imread(img_dir)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        transform_img = self.transform(image=img)['image'].astype(np.float32)
        img = transform_img.transpose(2, 0, 1)
        img = torch.tensor(img)

        if self.trainFlag:
            lable = self.pd.iloc[index]['individual_key']
            return img, lable
        else:
            return img

if __name__ == '__main__':
    pd = pd.read_csv('/archive/train_final.csv')
    dataset = HappyWhaleDataset(pd, True)
    dataloader = DataLoader(dataset, batch_size=32)
    img, label = next(iter(dataloader))
    print(img.shape)
