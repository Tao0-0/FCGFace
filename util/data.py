import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

class FaceDataset(ImageFolder):
    def __init__(self, data_root, transform=None):
        super(FaceDataset, self).__init__(data_root, transform)
    
    def __getitem__(self, idx):
        img_path = self.imgs[idx][0]
        label = self.imgs[idx][1]

        img = self.loader(img_path)
        img_name = os.path.basename(img_path)
        angle = img_name.split('.')[0]
        angle = float(angle.split('_')[1])/100
        if self.transform is not None:
            img = self.transform(img)
        return img, label, angle
