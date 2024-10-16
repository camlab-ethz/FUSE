import numpy as np
import torch
import pdb
import matplotlib.pyplot as plt

import sys
sys.path.append('../')
from utilities import *
        
    
def continuous_errors(pred, true):
    l1_errors = torch.zeros((128,))
    l2_errors = torch.zeros((128,))
    
    for sample in range(true.shape[0]):
        l1_errors[sample] = torch.mean(relative_error(pred[sample], true[sample], p=1)).item()
        l2_errors[sample] = torch.mean(relative_error(pred[sample], true[sample], p=2)).item()
        
        
    print(f"L1 error: {torch.mean(l1_errors).item()} +/- {torch.std(l1_errors).item()}")
    print(f"L2 error: {torch.mean(l2_errors).item()} +/- {torch.std(l2_errors).item()}")
    
    return

def continuous_crps(pred, true):
    crps_scores = np.zeros((128,))
    pred = pred.reshape(128,100,13*487)
    true = true.reshape(128,13*487)
    
    for sample in range(true.shape[0]):
        crps_score = crps_per_parameter(pred[sample], true[sample])
        crps_scores[sample] = np.mean(crps_score)
        
    print(f"Continuous CRPS: {np.mean(crps_scores).item()} +/- {np.std(crps_scores).item()}")
    
    
    
    return

def discrete_errors(pred, true):
    crps_scores = np.zeros((128,))
    
    for sample in range(true.shape[0]):
        crps_score = crps_per_parameter(pred[sample,:,:], true[sample])
        crps_scores[sample] = np.mean(crps_score)
        
    print(f"Discrete CRPS: {np.mean(crps_scores).item()} +/- {np.std(crps_scores).item()}")
    
    return
          
          
          
          
    
    
def main():
    n_train = 0
    n_test = 128
    n_val = 0
    batch_size = 1

    # True Parameters
    print('Using True Parameters')
    file_path = "../../_Data/cVANO_results/PWP/"
        # Load the compressed npz file
    data = np.load(file_path+'blood_flow_results_decoding_latent_vectors_of_true_parameters.npz')

    # Extract the arrays from the npz file
    true_pressures = data['true_pressures']
    pressures = data['pressures']
    true_parameters = data['parameters']
    
    true_pressures = torch.from_numpy(true_pressures)
    pressures = torch.from_numpy(pressures)
    true_parameters = torch.from_numpy(true_parameters)
    
    pressure_preds = pressures.permute(0,3,1,2)
    pressure_preds = pressure_preds[:,:,:,0]
    
    true_pressures = true_pressures.permute(0,2,1)
    
    true_parameters = true_parameters[:,0,:]
          
    continuous_errors(pressure_preds, true_pressures)
    
    for case in range(3):
        # Predicted Parameters
        print(f'Using Predicted Parameters Case {case}')
        file_path = "./"
            # Load the compressed npz file
        data = np.load(file_path+f'blood_flow_results_using_infered_parameters_case{case}.npz')


        # Extract the arrays from the npz file
        pressures = data['pressures']
        parameters = data['parameters']

        pressures = torch.from_numpy(pressures)
        parameters = torch.from_numpy(parameters)

        pressures = pressures.permute(0,2,3,1)
        pressure_preds = torch.mean(pressures, 1)
        

        
        continuous_errors(pressure_preds, true_pressures)

        parameters = parameters[:,:,0]

        discrete_errors(parameters, true_parameters)
        
          
    
    
    
    
    
    
    
    
    
    
if __name__ == "__main__":
    main()