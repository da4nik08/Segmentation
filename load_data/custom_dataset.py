from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os.path as osp
import torch
import torchvision
from torchvision.transforms import Resize
import cv2

def Create_mask(labels, img_size):
    mask = np.zeros(img_size*img_size, np.float32)
    if len(labels) > 1:
        np_lbl = np.asarray(labels, dtype=int)
        for i in range(0, len(np_lbl), 2):
            # Create a tuple of even and odd numbers
            start_ind = np_lbl[i]
            end_ind = np_lbl[i] + np_lbl[i + 1]
            mask[start_ind:end_ind] = 1.0

    return mask.reshape((img_size, img_size)).T

class CustomDataset(Dataset):
    def __init__(self, config):
        self.img_size = config['original_img_size']
        self.new_img_size = config['new_img_size']
        self.img_path = config['dataset']['train_img_path']
        self.labels = list(pd.read_csv(osp.join(config['dataset']['dir_path'], 
                                                'train_ship_segmentations_v2.csv'))["EncodedPixels"].fillna('').str.split())
        self.img_ids = list(pd.read_csv(osp.join(config['dataset']['dir_path'], 
                                                'train_ship_segmentations_v2.csv'))["ImageId"])
        self.resizer = Resize([self.new_img_size, self.new_img_size])

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        lbl = self.labels[idx]
        
        mask = Create_mask(lbl, self.img_size)                         # create mask
        
        image = cv2.imread(self.img_path+ "/" + img_id)                # load image
        res_im = self.resizer(torch.tensor(np.transpose(image, (2, 0, 1)))) # resize image to [3, 256, 256]
        norm_image = res_im / 127.5 - 1                                # normalization from -1 to 1
        #return image, mask
        return norm_image, self.resizer(torch.tensor(mask).unsqueeze(0)) # resize mask
