import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from timeit import default_timer
import pdb
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import yaml


import sys
sys.path.append('../')
from ood_dataset import LoadTurbTowers

sys.path.append('../../')
from utilities import *

sys.path.append('../../GAROM/')
import garom



seed = 0
torch.manual_seed(seed)

def eval_test_cases(model, test_loader, device, input_mins, input_maxs, p_min, p_max, n_points, n_test, pad):
    '''
    This function will generate several plots for each test case. These are:
    1. interesting C>C predictions
    '''
    evaluate_true_parameters = True
    calculate_errors = True
    plot_correlation = False
    plot_median_max = False
    box_plot_locations = False #  calculate_errors=True
    plot_fingerprint = False
    plot_joint_fingerprint =  False
    
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
    pad = 0
    n_test = 100
    path  = "../../_Data/ACB/"
    # Load the data from the configs
    loader = LoadTurbTowers()
    
    test_loader, input_mins, input_maxs, p_min, p_max, n_points = loader.get_dataloaders(path)

    if torch.cuda.is_available():
        device='cuda:0'
    
    
    model = torch.load("../../../_Models/ACB/GAROM_TurbTowers.pt").to(device)
        
    
    # Generate the plots
    eval_test_cases(model, test_loader, device, input_mins, input_maxs, p_min, p_max, n_points, n_test, pad)

    
    
if __name__ == "__main__":
    main()
    

