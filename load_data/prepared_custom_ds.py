from torch.utils.data import Dataset
import numpy as np
import torch
import cv2


class CustomDataset(Dataset):
    def __init__(self, config, img_ids, masks):
        self.img_path = config['dataset']['reshaped_img_path']
        self.img_ids = img_ids
        self.masks = masks

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        mask = self.masks[idx]                                                              # load mask        
        image = cv2.imread(self.img_path + "/" + img_id)                                    # load image
        tens_img = torch.tensor(np.transpose(image, (2, 0, 1)), dtype=torch.float32)        # create tensor image [3, 256, 256]
        norm_image = tens_img / 127.5 - 1                                                   # normalization from -1 to 1
        return norm_image, torch.tensor(mask, dtype=torch.float32).unsqueeze(0)             # return image, mask

