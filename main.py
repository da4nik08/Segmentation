import torch
from torch import nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
from models.small_model import Model_Unet
from utilities.config_load import load_config
from utilities.RecallAndPrecision import Metrics
import torchvision.transforms as transforms


def main(input_file, out_file):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    CONFIG_PATH = "configs/"
    config = load_config(CONFIG_PATH, "config.yaml")
    
    model = Model_Unet(kernel_size=config['model']['kernel_size'], 
                       dropout_rate=config['model']['dropout_rate'], 
                       nkernels=config['model']['nkernels'], 
                       output_chanels=config['model']['output_chanels'])
    model.to(device)
    model.load_state_dict(torch.load('model_svs/test_try_20240707_194916_8'))
    model.eval()

    image = cv2.imread(config['dataset']['reshaped_img_path'] + "/" + input_file)                                    
    tens_img = torch.tensor(np.transpose(image, (2, 0, 1)), dtype=torch.float32)
    transform = transforms.Compose([
        transforms.Resize((config['new_img_size'], config['new_img_size']))
    ])
    reshaped_img = transform(tens_img)
    with torch.inference_mode():
        pred = model(reshaped_img)
        
    image = torch.nn.functional.sigmoid(pred)
    image = image.numpy(force=True)
    image = np.transpose(image, (2, 0, 1))
    cv2.imwrite(config['dataset']['reshaped_img_path'] + "/" + out_file, image)
        