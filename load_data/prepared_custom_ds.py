from torch.utils.data import Dataset
from torchvision.transforms import Resize
import pandas as pd
import numpy as np
import os.path as osp
import torch
import torchvision
import cv2
from utilities.mask import Create_mask


class CustomDataset(Dataset):
    def __init__(self, config, img_ids, labels):
        self.img_size = config['original_img_size']
        self.new_img_size = config['new_img_size']
        self.img_path = config['dataset']['reshaped_img_path']
        self.mask_path = config['dataset']['mask_path']
        self.labels = labels
        self.img_ids = img_ids
        self.resizer = Resize([self.new_img_size, self.new_img_size])

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        lbl = self.labels[idx]

        mask = Create_mask(lbl, self.img_size)
        res_mask = self.resizer(torch.tensor(mask).unsqueeze(0))
        image = cv2.imread(self.img_path + "/" + img_id)
        image = torch.tensor(np.transpose(image, (2, 0, 1)), dtype=torch.float32)
        #image = torchvision.io.read_image(self.img_path + "/" + img_id).to(torch.float32)      # load image
        norm_image = image / 127.5 - 1                                                         # normalization from -1 to 1
        return norm_image, res_mask.type(torch.float32)
        #return norm_image, torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

