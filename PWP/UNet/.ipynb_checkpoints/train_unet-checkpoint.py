import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from timeit import default_timer
import pdb
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import yaml
import wandb
from unet_model import UNet
import sys

sys.path.append('../')
from dataset import LoadFullBody
from utilities import *


seed = 0
torch.manual_seed(seed)

l1_loss = torch.nn.L1Loss()
    
def train(model, train_loader, test_loader, device, config, p_max, p_min, param_max, param_min, noise_variance):
    # Training hyperparameters
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
    
    # Save a new model, start saving at some value to prevent saving many models at the beginning
    save_model = config['parameters']['save_model']['value']
    best_model = 2
    
    
    train_loss = 0
    for epoch in range(epochs):
        model.train()
        t1 = default_timer()
        iter = 0
        for theta, x, y in train_loader:
            theta = theta.to(device)
            channels = x.shape[1]
            mask = random_channel_mask(x.shape[0], input_channels=channels)
            x = x * mask.view(x.shape[0], channels, 1)
            
            x = x.to(device)
            y = y.to(device)

            # Must resize due to UNet restrictions to even points
            x = x[:,:,1:]
            y = y[:,:,1:]
            
            in_loss = model.param_loss(theta, x)
            
            optimizer.zero_grad()
            in_loss.backward()
            optimizer.step()
            
            pred = model.y_prediction(theta)
            out_loss = l1_loss(pred, y)
            
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
                times = x[:,-13:,:]
                channels = x.shape[1]
                mask = random_channel_mask(x.shape[0], input_channels=channels)
                x = x * mask.view(x.shape[0], channels, 1)
                
                x = x.to(device)
                y = y.to(device)

                # Resize to even input for UNet
                x = x[:,:,1:]
                y = y[:,:,1:]

                in_loss = model.param_loss(theta, x)
                in_test_loss_medians[epoch, iter] = in_loss.item()
                
                pred = model.y_prediction(theta)
                out_error = torch.mean(relative_error(pred, y))
                out_test_loss_medians[epoch, iter] = out_error.item()
                iter += 1

            scheduler.step(torch.median(in_test_loss_medians[epoch]))
            
            print(f'median test loss: {torch.median(in_test_loss_medians[epoch]).item():.4f} average test loss: {torch.mean(in_test_loss_medians[epoch]).item():.4f} +/- {torch.std(in_test_loss_medians[epoch]).item():.4f} ')
            print(f'median test error: {torch.median(out_test_loss_medians[epoch]).item():.4f} average test error: {torch.mean(out_test_loss_medians[epoch]).item():.4f}% +/- {torch.std(out_test_loss_medians[epoch]).item():.4f}% ')
            print('\n')
            
        test_error = torch.mean(in_test_loss_medians[epoch]).item()
        
        if save_model: 
            if test_error < best_model:
                print('******************')
                print(f'Saving model with error: {test_error:.4f}')
                torch.save(model, 'UNet_FullBody.pt')
                best_model = test_error
            
        
def main():
    with open("config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)
    
    # Get the data using a given normalization, select whether to use 
    # the parameters from Alastruey paper
    norm = config['parameters']['data_norm']
    parameter_subset = config['parameters']['params']
    n_train = config['parameters']['n_train']['value']
    n_val = config['parameters']['n_val']['value']
    n_test = config['parameters']['n_test']['value']
    batch_size = config['parameters']['batch_size']['value']
    
    # Load the data from the configs
    loader = LoadFullBody()
    train_loader, val_loader, test_loader, wv_shape, pm_shape, tg_shape, p_max, p_min, param_max, param_min = loader.get_dataloaders(n_train, n_val, n_test, batch_size, params = parameter_subset, norm=norm)
    
    # Define the model
    fmpe_layers = config['parameters']['fmpe_layers']['value']
    fmpe_width = config['parameters']['fmpe_width']['value']
    n_parameters = pm_shape[1]
    
    channels_in = wv_shape[1]
    channels_target = tg_shape[1]
    channels_latent = config['parameters']['latent_channels']['value']
    
    n_points = wv_shape[-1]
    if torch.cuda.is_available():
        device='cuda:0'
    else:
        device='cpu'
    
    model = UNet(n_parameters, n_points, fmpe_width, fmpe_layers, channels_in, channels_latent, channels_target).to(device)
    print(count_params(model))
    
    noise_variance = config['parameters']['noise_variance']['value']
    if noise_variance == 0:
        noise_variance = None
    train(model, train_loader, val_loader, device, config, p_max, p_min, param_max, param_min, noise_variance)
    
    
if __name__ == "__main__":
    main()
    
    
