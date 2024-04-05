from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os.path as osp
import torch
import torchvision
import cv2


class CustomDataset(Dataset):
    def __init__(self, config, img_ids, mask_ids):
        self.img_size = config['original_img_size']
        self.new_img_size = config['new_img_size']
        self.img_path = config['dataset']['train_img_path']
        self.mask_path = config['dataset']['mask_path']
        self.img_ids = img_ids
        self.mask_ids = mask_ids

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        mask_id = self.mask_ids[idx]
        mask = np.load(self.mask_path + "/" + mask_id)                                      # load mask        
        image = cv2.imread(self.img_path + "/" + img_id)                                    # load image
        tens_img = torch.tensor(np.transpose(image, (2, 0, 1)), dtype=torch.float32)        # create tensor image [3, 256, 256]
        norm_image = tens_img / 127.5 - 1                                                   # normalization from -1 to 1
        return norm_image, torch.tensor(mask, dtype=torch.float32).unsqueeze(0)             # return image, mask
