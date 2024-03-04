from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os.path as osp
import torch
import torchvision
import cv2


class CustomDataset(Dataset):
    def __init__(self, config):
        self.img_size = config['original_img_size']
        self.new_img_size = config['new_img_size']
        self.img_path = config['dataset']['reshaped_img_path']
        self.mask_path = config['dataset']['mask_path']
        self.labels = list(pd.read_csv(osp.join(config['dataset']['dir_path'], 
                                                'train_ship_segmentations_v2.csv'))["EncodedPixels"].fillna('').str.split())
        self.img_ids = list(pd.read_csv(osp.join(config['dataset']['dir_path'], 
                                                'train_ship_segmentations_v2.csv'))["ImageId"])

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        lbl = self.labels[idx]
        
        mask = torchvision.io.read_image(self.mask_path + "/mask" + img_id)                     # laod mask
        image = cv2.imread(self.img_path + "/" + img_id)
        image = torch.tensor(np.transpose(image, (2, 0, 1)), dtype=torch.float32)
        #image = torchvision.io.read_image(self.img_path + "/" + img_id).to(torch.float32)      # load image
        norm_image = image / 127.5 - 1                                                         # normalization from -1 to 1
        return norm_image, mask
