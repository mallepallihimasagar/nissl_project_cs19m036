import torch
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader
from skimage import io

torch.manual_seed(32)
np.random.seed(32)

Height=512
Width = 512

class Nissl_Dataset(Dataset):
    def __init__(self,root_dir='Nissl_Dataset',transforms=None,multiclass=True,cell_number=0):
        self.root_dir = root_dir
        self.transforms = transforms
        self.multiclass = multiclass
        self.cell_number = cell_number

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self,item):
        if torch.is_tensor(item):
            item = item.tolist()

        file_id = os.listdir(self.root_dir)[item]
        file_path = os.path.join(self.root_dir,file_id)
        # background = np.zeros((512,512),dtype=np.uint8)

        cell1_mask = cv2.imread(file_path+f'/{file_id}_mask1.png',cv2.IMREAD_GRAYSCALE)
        cell2_mask = cv2.imread(file_path+f'/{file_id}_mask2.png',cv2.IMREAD_GRAYSCALE)
        cell3_mask = cv2.imread(file_path+f'/{file_id}_mask3.png',cv2.IMREAD_GRAYSCALE)
        input_image = io.imread(file_path+f'/{file_id}.png')

        cell1_mask = cv2.resize(cell1_mask, (Width,Height),interpolation=cv2.INTER_NEAREST)
        cell2_mask = cv2.resize(cell2_mask, (Width,Height),interpolation=cv2.INTER_NEAREST)
        cell3_mask = cv2.resize(cell3_mask, (Width,Height),interpolation=cv2.INTER_NEAREST)
        input_image = cv2.resize(input_image,(Width,Height))
        
        # foreground = np.logical_or(cell1_mask,cell2_mask,cell3_mask)
        # background[np.where(foreground==False)] = 1
        foreground = np.zeros((Width,Height),dtype=np.uint8)
        if self.multiclass:
            foreground[cell1_mask==255]=1
            foreground[cell2_mask==255]=2
            foreground[cell3_mask==255]=3
        else :
            if self.cell_number==1:
                foreground[cell1_mask==255]=1
            elif self.cell_number==2:
                foreground[cell2_mask==255]=1
            elif self.cell_number==3:
                foreground[cell3_mask==255]=1

        if self.transforms:
            input_image = self.transforms(input_image)
            foreground = self.transforms(foreground)

        return input_image,foreground

            



        

