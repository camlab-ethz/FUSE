import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from timeit import default_timer
import pdb
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import yaml
import garom

import sys
sys.path.append('../')
from utilities import *
from dataset import LoadTurbTowers




seed = 0
torch.manual_seed(seed)

def eval_test_cases(model, test_loader, device, input_mins, input_maxs, p_min, p_max, n_points, n_test, pad):
    '''
    This function will generate several plots for each test case. These are:
    1. interesting C>C predictions
    '''
    evaluate_true_parameters = True
    calculate_errors = True
    
    crps_scores = np.zeros((n_test,))
    crps_all_scores = np.zeros((n_test,6))
    l1_errors = torch.zeros((n_test,20))
    l2_errors = torch.zeros((n_test,20))
    


    if evaluate_true_parameters:
        print('Using true parameters')
        if calculate_errors:
            with torch.no_grad(): 
                model.eval()
                
                iter = 0
                for theta, x, y in test_loader:
                    theta = theta.to(device)
                    y = y.to(device)
                    
                    y_pred = model(theta)
                    

                    y_mean = torch.mean(y_pred, 0)
                    l1_errors[iter] = relative_error(y_mean, y[0], p=1)
                    l2_errors[iter] = relative_error(y_mean, y[0], p=2)

                    iter+=1

                l1_error_means = torch.mean(l1_errors, 1)
                l2_error_means = torch.mean(l2_errors, 1)

                print(f'median L1 loss: {torch.median(l1_error_means).item():.4f} average L1 loss: {torch.mean(l1_error_means).item():.4f} +/- {torch.std(l1_error_means).item():.4f} ')
                print(f'max L1 loss: {torch.max(l1_error_means).item():.4f} ')
                print(f'median L2 loss: {torch.median(l2_error_means).item():.4f} average L2 loss: {torch.mean(l2_error_means).item():.4f} +/- {torch.std(l2_error_means).item():.4f} ')
                print(f'max L2 loss: {torch.max(l2_error_means).item():.4f} ')
                print('\n')


def main():
    with open("config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)
        
    
    # Parameters regarding the datasets
    n_train = config['parameters']['n_train']['value']
    n_test = config['parameters']['n_test']['value']
    n_val = config['parameters']['n_val']['value']
    batch_size = config['parameters']['batch_size']['value']
    pad = config['parameters']['pad']['value']
    data_path = config['parameters']['data_path']
    
    # Load the data from the configs
    loader = LoadTurbTowers()
    train_loader, val_loader, test_loader, input_mins, input_maxs, p_min, p_max, n_points = loader.get_dataloaders(n_train, n_val, n_test, batch_size, data_path)
    
    
    if torch.cuda.is_available():
        device='cuda:0'
    
    
    # model = torch.load("../../_Models/ACB/GAROM_TurbTowers.pt").to(device)
    model = torch.load("/cluster/work/math/camlab-data/FUSE/_Models/ACB/GAROM_TurbTowers.pt").to(device)
        
    
    # Generate the plots
    eval_test_cases(model, test_loader, device, input_mins, input_maxs, p_min, p_max, n_points, n_test, pad)

    
    
if __name__ == "__main__":
    main()
    

