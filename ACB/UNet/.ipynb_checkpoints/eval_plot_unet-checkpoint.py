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
from dataset import LoadTurbTowers


    


seed = 0
torch.manual_seed(seed)

def eval_test_cases(model, test_loader, device, input_mins, input_maxs, p_min, p_max, n_points, n_test, pad):
    '''
    This function will generate several plots for each test case. These are:
    1. interesting C>C predictions
    '''
    evaluate_full_prediction = True
    evaluate_true_parameters = True
    calculate_errors = True
    plot_correlation = False
    plot_median_max = True
    box_plot_locations = False #  calculate_errors=True
    plot_fingerprint = False
    plot_joint_fingerprint =  False
    
    crps_scores = np.zeros((n_test,))
    crps_all_scores = np.zeros((n_test,6))
    l1_errors = torch.zeros((n_test,20))
    l2_errors = torch.zeros((n_test,20))
    continuous_crps = torch.zeros((n_test,))
    
        
    if evaluate_full_prediction:
        print('Using predicted parameters')
        if calculate_errors:
            with torch.no_grad(): 
                model.eval()
                # EVALUATE THE ERRORS OVER THE TEST SET
                iter = 0
                for theta, x, y in test_loader:
                    theta = theta.to(device)

                    x = x[:,:,1:]
                    y = y[:,:,1:]
                        
                    x = x.to(device)

                    y = y.to(device)
                    
                    theta_pred, y_pred = model.full_flow(x, n_samples=100)


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

                print(f'median crps: {np.median(crps_scores):.4f} average crps: {np.mean(crps_scores):.4f} +/- {np.std(crps_scores):.4f} ')
                print(f'median L1 loss: {torch.median(l1_error_means).item():.4f} average L1 loss: {torch.mean(l1_error_means).item():.4f} +/- {torch.std(l1_error_means).item():.4f} ')
                print(f'max L1 loss: {torch.max(l1_error_means).item():.4f} ')
                print(f'median L2 loss: {torch.median(l2_error_means).item():.4f} average L2 loss: {torch.mean(l2_error_means).item():.4f} +/- {torch.std(l2_error_means).item():.4f} ')
                print(f'max L2 loss: {torch.max(l2_error_means).item():.4f} ')
                print('\n')
            
                # find the location of the median and maximum
                crps_median = np.median(crps_scores)
                crps_median_index = np.abs(crps_scores-crps_median).argmin()
                crps_max_index = crps_scores.argmax()
                l1_max_index = l1_error_means.argmax()
                print(crps_median_index)
                print(crps_max_index)
                print(l1_max_index)
                
                if box_plot_locations:
                    box_per_parameter(crps_all_scores, 'C_D/box_plots.pdf')
                    box_per_location(l1_errors, 'C_C/box_plots.pdf')
                

                
        else:
            # these are known, we don't have to test the model every time
            crps_median_index = 891
            crps_max_index = 755
            l1_max_index = 221
        interesting_index = 2




        # PLOT THE MEDIAN AND MAX SAMPLES
        if plot_median_max:
            iter = 0
            for theta, x, y in test_loader:
                theta = theta.to(device)
                
                x = x[:,:,1:]
                y = y[:,:,1:]
                    
                x = x.to(device)
                y = y.to(device)

                if iter == crps_median_index:
                    make_some_plots(x, y, theta, model, input_mins, input_maxs, p_min, p_max, pad, 'median')
                elif iter == crps_max_index:
                    make_some_plots(x, y, theta, model, input_mins, input_maxs, p_min, p_max, pad, 'crps_max')
                elif iter == l1_max_index:
                    make_some_plots(x, y, theta, model, input_mins, input_maxs, p_min, p_max, pad, 'l1_max')
                elif iter == interesting_index:
                    make_some_plots(x, y, theta, model, input_mins, input_maxs, p_min, p_max, pad, 'interesting')

                iter +=1
                

    if evaluate_true_parameters:
        print('Using true parameters')
        if calculate_errors:
            with torch.no_grad(): 
                model.eval()
                
                iter = 0
                for theta, x, y in test_loader:
                    theta = theta.to(device)
                             
                        
                    x = x[:,:,1:]
                    y = y[:,:,1:]

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
            
                # find the location of the median and maximum
                l1_median = np.median(l1_error_means)
                l1_median_index = np.abs(l1_error_means-l1_median).argmin()
                l1_max_index = l1_error_means.argmax()
                print(l1_median_index)
                print(l1_max_index)
                
                if box_plot_locations:
                    box_per_location(l1_errors, 'True_Parameters/box_plots.pdf')
                
                
        else:
            # these are known, we don't have to test the model every time
            l1_median_index = 587
            l1_max_index = 221
        interesting_index = 2




        # PLOT THE MEDIAN AND MAX SAMPLES
        if plot_median_max:
            iter = 0
            for theta, x, y in test_loader:
                theta = theta.to(device)
                y = y.to(device)

                if iter == l1_median_index:
                    make_some_plots_true(y, theta, model, input_mins, input_maxs, pad, 'median')
                elif iter == l1_max_index:
                    make_some_plots_true(y, theta, model, input_mins, input_maxs, pad, 'max')
                elif iter == interesting_index:
                    make_some_plots_true(y, theta, model, input_mins, input_maxs, pad, 'interesting')

                iter +=1
                    


    # PLOT FINGERPRINTS FOR THE SELECTED PARAMETERS
    if plot_fingerprint:
        indices = [0, 1, 2, 3, 4, 5]
        print("Plotting Fingerprints...")
        for param_id in indices:
            plot_fingerprints(model, input_mins, input_maxs, p_min, p_max, param_id)
        plt.close('all')

    if plot_joint_fingerprint:
        print("Plotting Joints...")
        joint_fingerprint(model, input_mins, input_maxs, p_min, p_max)
    
    

        
def main():
    with open("config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)
        
    n_train = config['parameters']['n_train']['value']
    n_val = config['parameters']['n_val']['value']
    n_test = config['parameters']['n_test']['value']
    batch_size = config['parameters']['batch_size']['value']
    pad = config['parameters']['pad']['value']
    
    # Load the data from the configs
    loader = LoadTurbTowers()
    
    
    train_loader, val_loader, test_loader, input_mins, input_maxs, p_min, p_max, n_points = loader.get_dataloaders(n_train, n_val, n_test, batch_size)
    
    point_reduction = 1
    n_points = int(np.ceil(n_points/point_reduction))
    if torch.cuda.is_available():
        device='cuda:0'
            
            
    model = torch.load('../../_Models/ACB/UNet_TurbTowers.pt').to(device)
        
    
    # Generate the plots
    eval_test_cases(model, test_loader, device, input_mins, input_maxs, p_min, p_max, n_points, n_test, pad)

    
    
if __name__ == "__main__":
    main()
    

