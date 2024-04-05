import numpy as np


def crop3x3channel(img, i, size):
    return img[(i//3)*size: ((i//3)+1)*size, (i%3)*size: (i%3+1)*size, :]

def crop3x3(img, i, size):
    return img[(i//3)*size: ((i//3)+1)*size, (i%3)*size: (i%3+1)*size]

def crop3x3_mask(img, numb_obj):
    sums = np.zeros((9), dtype=float)
    for i in range(9):
        sums[i] = np.sum(crop3x3(img, i, config['new_img_size']))
        
    sort_arg = np.argsort(sums)
    num_crops = numb_obj
    
    for sum in np.sort(sums)[-numb_obj:]:
        if sum < 1:
            num_crops -= 1
            
    return sort_arg[-num_crops:]