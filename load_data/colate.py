import torch
import numpy as np
from utilities.mask import Create_mask

def process_data():
    
    return numpy.array

def collate_fn(batch):
    images_batch, labels_batch = zip(*batch)

    mask_batch = []
    for label in labels_batch:
        processed_data = torch.tensor(process_data(label), dtype=torch.float)
        processed_batch.append(processed_data)
    return torch.stack(processed_batch), torch.tensor(labels_batch)