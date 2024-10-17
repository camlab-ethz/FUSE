import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from timeit import default_timer
import pdb
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import yaml
from unet_model import UNet

import sys
sys.path.append('../')
from utilities import *
from dataset import LoadFullBody


seed = 0
torch.manual_seed(seed)

l1_loss = torch.nn.L1Loss(reduction='none')

def eval_test_cases(model, test_loader, device, p_max, p_min, param_max, param_min, point_reduction, n_test):

    evaluate_test_cases = True
    evaluate_true_case = True

    # 1: perfect information
    test_case_1 = torch.ones(52)
    
    # 2: pressure, velocity, PPG at wrist
    test_case_2 = torch.zeros(52)
    # wrist
    test_case_2[9] = 1       # pressure
    test_case_2[13+9] = 1    # velocity
    test_case_2[2*13+9] = 1  # PPG
    test_case_2[3*13+9] = 1  # measurement times


    # 3: PPG at finger
    test_case_3 = torch.zeros(52)
    # finger
    test_case_3[2*13+6] = 1  # PPG
    test_case_3[3*13+6] = 1  # measurement times
    
    
    test_cases = [test_case_1, test_case_2, test_case_3]
    
    n_test_cases = len(test_cases)
    crps_scores = np.zeros((n_test,))
    crps_per_case = np.zeros((n_test,3))
    l1_errors = torch.zeros((n_test,13))
    l2_errors = torch.zeros((n_test,13))
    l1_errors_per_case = torch.zeros((n_test,4))
    
    if evaluate_test_cases:
        with torch.no_grad(): 
            model.eval()
            for case, mask in enumerate(test_cases):
                
                # EVALUATE THE ERRORS OVER THE TEST SET
                iter = 0
                for theta, x, y in test_loader:
                    theta = theta.to(device)
                    times = x[:,-13:,:]
                    
                    times = times[:,:,1:]
                    x = x[:,:,1:]
                    y = y[:,:,1:]
                        
                    x = x * mask.view(1, x.shape[1], 1)
                    x = x[:,:,::point_reduction].to(device)
                    y = y[:,:,::point_reduction].to(device)
                    

                    theta_pred, y_pred = model.full_flow(x, n_samples=100)

                    y_mean = torch.mean(y_pred, 0)
                    crps_score = crps_per_parameter(theta_pred[:,0,:], theta[0])

                    crps_scores[iter] = np.mean(crps_score)
                    crps_per_case[iter, case] = np.mean(crps_score)
                    l1_errors[iter] = relative_error(y_mean, y[0], p=1)
                    l2_errors[iter] = relative_error(y_mean, y[0], p=2)
                    l1_errors_per_case[iter, case+1] = torch.mean(relative_error(y_mean, y[0], p=1))
                    iter+=1

                l1_error_means = torch.mean(l1_errors, 1)
                l2_error_means = torch.mean(l2_errors, 1)
                print('\n')
                print(f'Test Case {case}')
                print(f'median crps: {np.median(crps_scores):.4f} average crps: {np.mean(crps_scores):.4f} +/- {np.std(crps_scores):.4f} ')
                print(f'median L1 loss: {torch.median(l1_error_means).item():.4f} average L1 loss: {torch.mean(l1_error_means).item():.4f} +/- {torch.std(l1_error_means).item():.4f} ')
                print(f'max L1 loss: {torch.max(l1_error_means).item():.4f} ')
                print(f'median L2 loss: {torch.median(l2_error_means).item():.4f} average L2 loss: {torch.mean(l2_error_means).item():.4f} +/- {torch.std(l2_error_means).item():.4f} ')
                print(f'max L2 loss: {torch.max(l2_error_means).item():.4f} ')
                print('\n')

                    
    if evaluate_true_case:
        # EVALUATE THE ERRORS OVER THE WAVEFORMS GIVEN THE TRUE PARAMETERS
        with torch.no_grad(): 
            model.eval()
            iter = 0
            for theta, x, y in test_loader:
                theta = theta.to(device)
                
                y = y[:,:,1:]
                    
                y_pred = model.y_prediction(theta, n_points=y.shape[-1])
                
                y_pred = y_pred.cpu()
                

                    
                l1_errors[iter] = relative_error(y_pred[0], y[0], p=1)
                l2_errors[iter] = relative_error(y_pred[0], y[0], p=2)
                iter+=1
                
            # find the location of the median and maximum
            l1_error_means = torch.mean(l1_errors, 1)
            l1_errors_per_case[:,0] = l1_error_means
            l2_error_means = torch.mean(l2_errors, 1)
            error_median = torch.median(l1_error_means)
            error_median_index = torch.abs(l1_error_means-error_median).argmin()
            error_max_index = l1_error_means.argmax()
            
                
            print(f'Using True Parameters')
            print(f'median L1 loss: {torch.median(l1_error_means).item():.4f} average L1 loss: {torch.mean(l1_error_means).item():.4f} +/- {torch.std(l1_error_means).item():.4f} ')
            print(f'max L1 loss: {torch.max(l1_error_means).item():.4f} ')
            print(f'median L2 loss: {torch.median(l2_error_means).item():.4f} average L2 loss: {torch.mean(l2_error_means).item():.4f} +/- {torch.std(l2_error_means).item():.4f} ')
            print(f'max L2 loss: {torch.max(l2_error_means).item():.4f} ')
            print('\n')
            
        return 

        
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
    
    
    point_reduction = 1
    n_points = int(np.ceil(wv_shape[1]/point_reduction))
    
    if torch.cuda.is_available():
        device='cuda:0'
    else:
        device='cpu'
        
    # Load the model
    model = torch.load('../../_Models/PWP/UNet_FullBody.pt')
    
       
    print(count_params(model))
    eval_test_cases(model, test_loader, device, p_max, p_min, param_max, param_min, point_reduction, n_test)

    
    
if __name__ == "__main__":
    main()
    

