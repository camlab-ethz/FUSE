import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import yaml

from FUSE import FUSE

import sys
sys.path.append('../')
from utilities import *
from dataset import LoadTurbTowers


seed = 0
torch.manual_seed(seed)

def eval_test_cases(model, test_loader, device, input_mins, input_maxs, p_min, p_max, n_points, n_test, pad):

    # Define the problems which we would like to evaluate
    evaluate_full_prediction = True      # Inverse & Forward pass through full model
    evaluate_true_parameters = True      # Forward pass only, takes parameters 
    calculate_errors = True
    
    crps_scores = np.zeros((n_test,))
    crps_all_scores = np.zeros((n_test,6))
    l1_errors = torch.zeros((n_test,20))
    l2_errors = torch.zeros((n_test,20))
    
    for sample_power in range(7, 8):
        n_samples = 2**sample_power
        print(f"{n_samples} samples")
        if evaluate_full_prediction:
            print('Using predicted parameters')
            if calculate_errors:
                with torch.no_grad(): 
                    model.eval()
                    # EVALUATE THE ERRORS OVER THE TEST SET
                    iter = 0
                    for theta, x, y in test_loader:

                        if pad >0:
                            x = extrapolate_zeros(x, pad)
                            
                        x = x.to(device)
                        y = y.to(device)
                        theta = theta.to(device)

                        theta_pred, y_pred = model.full_flow(x, n_samples=n_samples)

                        if pad > 0:
                            y_pred = y_pred[:,:,pad:-pad]

                        crps_score = crps_per_parameter(theta_pred[:,0,:], theta[0])
                        crps_all_scores[iter] = crps_score
                        crps_scores[iter] = np.mean(crps_score)
                        
                        y_mean = torch.mean(y_pred, 0)
                        l1_errors[iter] = relative_error(y_mean, y[0], p=1)
                        l2_errors[iter] = relative_error(y_mean, y[0], p=2)

                        iter+=1

                    l1_error_means = torch.mean(l1_errors, 1)
                    l2_error_means = torch.mean(l2_errors, 1)
                    
                    print(np.mean(crps_all_scores, 0))
                    print(np.std(crps_all_scores, 0))

                    print(f'median parameter crps: {np.median(crps_scores):.4f} average crps: {np.mean(crps_scores):.4f} +/- {np.std(crps_scores):.4f} ')
                    print(f'median L1 loss: {torch.median(l1_error_means).item():.4f} average L1 loss: {torch.mean(l1_error_means).item():.4f} +/- {torch.std(l1_error_means).item():.4f} ')
                    print(f'max L1 loss: {torch.max(l1_error_means).item():.4f} ')
                    print(f'median L2 loss: {torch.median(l2_error_means).item():.4f} average L2 loss: {torch.mean(l2_error_means).item():.4f} +/- {torch.std(l2_error_means).item():.4f} ')
                    print(f'max L2 loss: {torch.max(l2_error_means).item():.4f} ')
                    print('\n')
               
    if evaluate_true_parameters:
        print('Using true parameters')
        if calculate_errors:
            with torch.no_grad(): 
                model.eval()
                
                iter = 0
                for theta, x, y in test_loader:
                    theta = theta.to(device)
                    y = y.to(device)

                    y_pred = model.y_prediction(theta, n_points = y.shape[-1]+2*pad)
                    
                    if pad > 0:
                        y_pred = y_pred[:,:,pad:-pad]

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
    data_path = config['parameters']['data_path']
    pad = config['parameters']['pad']['value']
    
    # Load the data from the configs
    loader = LoadTurbTowers()
    train_loader, val_loader, test_loader, input_mins, input_maxs, p_min, p_max, n_points = loader.get_dataloaders(n_train, n_val, n_test, batch_size, data_path)
        
    point_reduction = 1
    n_points = int(np.ceil(n_points/point_reduction))
    
    if torch.cuda.is_available():
        device='cuda:0'
        model = torch.load('../../_Models/ACB/FUSE_TurbTowers.pt').to(device)
        
    # Generate the plots
    eval_test_cases(model, test_loader, device, input_mins, input_maxs, p_min, p_max, n_points, n_test, pad)

    
    
    
if __name__ == "__main__":
    main()
    

