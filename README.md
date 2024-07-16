
# Ship Detection Unet

Airbus Ship Detection Challenge was decided to be solved using model Unet.



## Description development features

### Preprocessing

The pictures were divided into 9 parts of 3 by 3, and those that contained more ships were selected. This was done due to the large class imbalance on the masks. The last step was to create the masks. All this data stored in directory **dataset**.

### Model

The small UNET model was created to solve the segmentation problem. It is stored in **models**.

### Loss function

To get rid of the problem of class imbalance, weighted cross entropy function was developed.

### Training

A small model was trained and obtained normal results, but with the increase of the model and solving the overheating problem, the model result can be significantly improved.
```
LOSS test 0.0040710256434977055
Recall test 0.7976382970809937
Precision test 0.7942672967910767
Test TP->4538703.0 | FN ->1151474.0| FP->1175624.0 | TN->212483184.0
Dice test 0.7959492144507118
IoU test 0.6610594987869263
```

The weight tensors are saved in **model_svs**
## Usage inference

The image must be in the directory **dataset/inference_img** (or change settings in config file)
- first argument is name of input image
- second argument is name of result image

```bash
inference.py input_file output_file
```


## Tech Stack

**Model:** Pytorch

**Data:** Pandas, Numpy

