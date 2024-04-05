import numpy as np

def Create_mask(labels, img_size):
    mask = np.zeros(img_size*img_size, np.int8)
    if len(labels) > 1:
        np_lbl = np.asarray(labels, dtype=int)
        for i in range(0, len(np_lbl), 2):
            # Create a tuple of even and odd numbers
            start_ind = np_lbl[i]
            end_ind = np_lbl[i] + np_lbl[i + 1]
            mask[start_ind:end_ind] = 1

    return mask.reshape((img_size, img_size)).T