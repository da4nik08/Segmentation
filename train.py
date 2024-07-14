from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from load_data.prepared_custom_ds import CustomDataset
from utilities.config_load import load_config
from utilities.RecallAndPrecision import Metrics
from utilities.weighted_cross_entropy import weighted_cross_entropy
import time
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from sklearn.model_selection import train_test_split
import os.path as osp
import pandas as pd
import os


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CONFIG_PATH = "configs/"


config = load_config(CONFIG_PATH, "config.yaml")
labels = list(pd.read_csv(osp.join(config['dataset']['dir_path'], 
                                                'train_ship_segmentations_v2.csv'))["EncodedPixels"].fillna('').str.split())
img_ids = list(pd.read_csv(osp.join(config['dataset']['dir_path'], 
                                                'train_ship_segmentations_v2.csv'))["ImageId"])


masks = list()
for id in tqdm(mask_ids):
    masks.append(np.load(config['dataset']['mask_path'] + "/" + id))

X_train, X_test, y_train, y_test = train_test_split(img_ids, 
                                                    masks, 
                                                    test_size=0.05,
                                                    random_state=42)


def train_step(model, loss_fn, opt, loader):
    loss_per_batches = 0
    elapsed = 0
    start_epoch2 = time.time()
    for i, data in tqdm(enumerate(loader), total=66922//72):

        start_epoch = time.time()
        features, labels = data
        features, labels = features.to(device), labels.to(device)
        opt.zero_grad()
        
        y_pred = model(features)
        loss, _ = loss_fn(torch.flatten(y_pred), torch.flatten(labels))
        loss.backward()
        
        opt.step()
        
        loss_per_batches += loss
        end_epoch = time.time()
        elapsed += (end_epoch - start_epoch)

    print("train = " + str(elapsed))
    print("train + load = " + str(time.time() - start_epoch2))
    return loss_per_batches/(i+1)

def train(model, loss_fn, opt, train_loader, val_loader, save_treshold=10, epochs=50, model_name='model_name'):
        
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/' + model_name + '_{}'.format(timestamp))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=3, verbose=True)
    
    for epoch in range(epochs):
        start_epoch = time.time()
        metrics_valid = Metrics()
        print('EPOCH {}:'.format(epoch + 1))
        
        model.train()
        avg_loss = train_step(model, loss_fn, opt, train_loader)
        model.eval()

        vloss = 0
        counter = 0
        with torch.inference_mode():
            for i, vdata in enumerate(val_loader):
                vfeatures, vlabels = vdata
                vfeatures, vlabels = vfeatures.to(device), vlabels.to(device)

                y_pred = model(vfeatures)
                bloss, y_pred = loss_fn(torch.flatten(y_pred), torch.flatten(vlabels))
                vloss += bloss
                metrics_valid.batch_step(torch.flatten(vlabels), y_pred)
                counter = i

        avg_vloss = vloss / (counter + 1)
        
        scheduler.step(avg_loss)

        valrecall, valprecision, valmetr = metrics_valid.get_metrics()
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
        print('Recall valid {}'.format(valrecall))
        print('Precision valid {}'.format(valprecision))
        print('Val TP->{} | FN ->{}| FP->{} | TN->{}'.format(*valmetr))
        
        writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch + 1)
        writer.add_scalars('Validation Metrics',
                    { 'Validation Recall' : valrecall, 'Training Precision' : valprecision
                    }, epoch + 1)
        
        if (epoch + 1) % save_treshold == 0:
            model_path = config['model']['svs_path'] + model_name +'_{}_{}'.format(timestamp, (epoch + 1))
            torch.save(model.state_dict(), model_path)
        end_epoch = time.time()
        elapsed = end_epoch - start_epoch
        print("Time per epoch {}s".format(elapsed))


dataset = CustomDataset(config, X_train, y_train)
vdataset = CustomDataset(config, X_test, y_test)
dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=config['dataloader']['batch_size'],
                                         num_workers=config['dataloader']['num_workers'], 
                                         shuffle=config['dataloader']['shuffle'])
vdataloader = torch.utils.data.DataLoader(vdataset,
                                         batch_size=config['dataloader']['batch_size'],
                                         num_workers=config['dataloader']['num_workers'], 
                                         shuffle=config['dataloader']['shuffle'])


model = Model_Unet(kernel_size=config['model']['kernel_size'], 
                   dropout_rate=config['model']['dropout_rate'], 
                   nkernels=config['model']['nkernels'], 
                   output_chanels=config['model']['output_chanels'])
loss_fn = weighted_cross_entropy
optimizer = torch.optim.Adam(model.parameters(),
                             lr=config['Adam']['learning_rate'], 
                             betas=(config['Adam']['beta1'], config['Adam']['beta2']), 
                             eps=float(config['Adam']['epsilon']))
model.to(device)

train(model, loss_fn, optimizer, dataloader, vdataloader, 
      save_treshold=config['train']['save_treshold'], 
      epochs=config['train']['epochs'], 
      model_name=config['train']['model_name'])