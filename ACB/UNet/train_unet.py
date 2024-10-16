import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from timeit import default_timer
import pdb
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from unet_model import UNet
import yaml
import wandb

import sys
sys.path.append('../')

from utilities import *
from dataset import LoadTurbTowers


seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

l1_loss = torch.nn.L1Loss()
nll_loss = torch.nn.GaussianNLLLoss()
    
def train(model, train_loader, test_loader, device, config, input_mins, input_maxs, p_min, p_max):
    
    lr = config['parameters']['learning_rate']['value']
    batch_size = config['parameters']['batch_size']['value']
    wd = config['parameters']['weight_decay']['value']
    factor = config['parameters']['factor']['value']
    patience = config['parameters']['patience']['value']

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, verbose=True)
    
    epochs = config['parameters']['epochs']['value']
    n_train = config['parameters']['n_train']['value']
    n_test = config['parameters']['n_test']['value']
    train_loss_in = torch.zeros(int(np.ceil(n_train/batch_size)))
    train_loss_out = torch.zeros(int(np.ceil(n_train/batch_size)))
    in_test_loss_medians = torch.zeros(epochs, n_test)
    out_test_loss_medians = torch.zeros(epochs, n_test)
    
    # pad = config['parameters']['pad']['value']
    pad = 0
    best_model = 15
    
    train_loss = 0
    for epoch in range(epochs):
        model.train()
        t1 = default_timer()
        iter = 0
        for theta, x, y in train_loader:
            theta = theta.to(device)
            x = x.to(device)
            y = y.to(device)
            
            # set to be 180 points; can't go back to 181 with U-Net
            x = x[:,:,1:]
            y = y[:,:,1:]
            
            
            in_loss = model.param_loss(theta, x)
            
            optimizer.zero_grad()
            in_loss.backward()
            optimizer.step()
            
            pred = model.y_prediction(theta)
            out_loss = l1_loss(pred, y) + 100 * h1_loss(pred, y)
            
            optimizer.zero_grad()
            out_loss.backward()
            optimizer.step()
            
            train_loss_in[iter] = in_loss.item()
            train_loss_out[iter] = out_loss.item()

            
            iter += 1
        t2 = default_timer()
        
        print(f'epoch: {epoch} \t time: {t2-t1}')
        print(f'median train loss: {torch.median(train_loss_in).item():.4f} average train loss: {torch.mean(train_loss_in).item():.4f} +/- {torch.std(train_loss_in).item():.4f} ')
        print(f'median train loss: {torch.median(train_loss_out).item():.4f} average train loss: {torch.mean(train_loss_out).item():.4f} +/- {torch.std(train_loss_out).item():.4f} ')

        with torch.no_grad():   
            model.eval()
            test_loss = 0
            iter = 0
            for theta, x, y in test_loader:
                theta = theta.to(device)

                # set to be 180 points; can't go back to 181 with U-Net
                x = x[:,:,1:]
                y = y[:,:,1:]
            
                x = x.to(device)
                y = y.to(device)
                
                in_loss = model.param_loss(theta, x)
                
                pred = model.y_prediction(theta)
                out_error = relative_l1_error(pred, y)
                
                in_test_loss_medians[epoch, iter] = in_loss.item()
                out_test_loss_medians[epoch, iter] = out_error.item()
                iter += 1

            
            scheduler.step(torch.median(in_test_loss_medians))
            
            print(f'median test loss: {torch.median(in_test_loss_medians[epoch]).item():.4f} average test loss: {torch.mean(in_test_loss_medians[epoch]).item():.4f} +/- {torch.std(in_test_loss_medians[epoch]).item():.4f} ')
            print(f'median test error: {torch.median(out_test_loss_medians[epoch]).item():.4f} average test error: {torch.mean(out_test_loss_medians[epoch]).item():.4f}% +/- {torch.std(out_test_loss_medians[epoch]).item():.4f}% ')
            print('\n')
            
                        
        test_error = torch.mean(out_test_loss_medians[epoch]).item()
        if test_error < best_model:
            print('******************')
            print(f'Saving model with error: {test_error:.4f}')
            torch.save(model, 'UNet_TurbTowers.pt')
            best_model = test_error
            

def main():
    with open("config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)
        
    # Parameters regarding the datasets
    n_train = config['parameters']['n_train']['value']
    n_test = config['parameters']['n_test']['value']
    n_val = config['parameters']['n_val']['value']
    batch_size = config['parameters']['batch_size']['value']
    data_path = config['parameters']['data_path']
    
    # Load the data from the configs
    loader = LoadTurbTowers()
    train_loader, val_loader, test_loader, input_mins, input_maxs, p_min, p_max, n_points = loader.get_dataloaders(n_train, n_val, n_test, batch_size, data_path)
    
    # Define the model
    n_parameters = p_min.shape[0]    
    channels_in = input_mins.shape[1]
    channels_target = channels_in
    channels_latent = config['parameters']['channels_latent']['value']
    fmpe_layers = config['parameters']['fmpe_layers']['value']
    fmpe_width = config['parameters']['fmpe_width']['value']
    
        
    if torch.cuda.is_available():
        device='cuda:0'
    else:
        device='cpu'
    
    
    model = UNet(n_parameters, fmpe_width, fmpe_layers, channels_in, channels_latent, channels_target)
    print(count_params(model))
    model.to(device)  

    noise_variance = config['parameters']['noise_variance']['value']
    if noise_variance == 0:
        noise_variance = None
    train(model, train_loader, val_loader, device, config, input_mins, input_maxs, p_min, p_max)
    
    
if __name__ == "__main__":
    main()
    
    
