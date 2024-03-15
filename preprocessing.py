import numpy as np
import pandas as pd
import os
import yaml
import cv2
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision.transforms import Resize
import time
import os.path as osp
from tqdm import tqdm
from utilities.config_load import load_config

def preprocessing():
    CONFIG_PATH = "configs/"
    config = load_config("config.yaml")

    train_ship_data = pd.read_csv("dataset/train_ship_segmentations_v2.csv")
    labels = list(pd.read_csv(osp.join(config['dataset']['dir_path'], 
                                                'train_ship_segmentations_v2.csv'))["EncodedPixels"].fillna('').str.split())
    img_ids = list(pd.read_csv(osp.join(config['dataset']['dir_path'], 
                                                'train_ship_segmentations_v2.csv'))["ImageId"])
    resizer = Resize([config['new_img_size'], config['new_img_size']])

    for img_id in tqdm(img_ids):
        image = cv2.imread(config['dataset']['train_img_path'] + "/" + img_id)
        res_im = resizer(torch.tensor(np.transpose(image, (2, 0, 1))))
        perm_im = res_im.permute(1, 2, 0)
        cv2.imwrite(config['dataset']['reshaped_img_path'] + "/" + img_id, np.uint8(perm_im.numpy()))