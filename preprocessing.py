import numpy as np
import pandas as pd
import os
import yaml
import cv2
import pickle
import os.path as osp
from tqdm import tqdm
from utilities.config_load import load_config
from utilities.crop_functions import *
from utilities.pkl import save_pkl


def preprocessing():
    CONFIG_PATH = "configs/"
    config = load_config("config.yaml")

    train_ship_data = pd.read_csv("dataset/train_ship_segmentations_v2.csv")
    img_ids = list(pd.read_csv(osp.join(config['dataset']['dir_path'], 
                                                'train_ship_segmentations_v2.csv'))["ImageId"])

    list_ids = list()
    for img_id in tqdm(list(set(img_ids))):
        tmask = np.zeros((config['original_img_size'], config['original_img_size']), np.int8)
        numb_obj = 0
        for label in list(train_ship_data[train_ship_data['ImageId']==img_id]['EncodedPixels'].fillna('').str.split()):
            if len(label) < 2:
                continue
            mask = Create_mask(label, config['original_img_size'])
            tmask = tmask | mask
            numb_obj += 1
        
        if int(np.sum(tmask)) == 0:
            continue
    
        image = cv2.imread(config['dataset']['train_img_path'] + "/" + img_id)
        crops_index = crop3x3_mask(tmask, numb_obj)
        for ind in crops_index:
            np.save(config['dataset']['mask_path'] + '/' + str(ind) + img_id , crop3x3(tmask, ind, config['new_img_size']))
            cv2.imwrite(config['dataset']['reshaped_img_path'] + '/' + str(ind) + img_id, 
                        crop3x3channel(image, ind, config['new_img_size']))
            list_ids.append(str(ind) + img_id)

    save_pkl('dataset/listofindex.pkl', list_ids)