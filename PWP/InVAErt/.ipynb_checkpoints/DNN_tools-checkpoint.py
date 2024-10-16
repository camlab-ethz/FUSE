# Toolkit for all kinds of network

import os.path
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset,  DataLoader
from matplotlib import pyplot as plt
import matplotlib
from skspatial.objects import Line, Points, Plane
from numpy import linalg as LA


#-----------------------------------------------------------------#
# auto-batching tools for NF model
class MyDatasetX(Dataset):
    def __init__(self, X):
        super(MyDatasetX, self).__init__()        
        self.X = X

    # number of samples to be batched
    def __len__(self):
        return self.X.shape[0] 
       
    # get samples
    def __getitem__(self, index):
        return self.X[index]
#------------------------------------------------------------------#


#-----------------------------------------------------------#
# auto-batching tools
class MyDatasetXY(Dataset):
    def __init__(self, X, Y):
        super(MyDatasetXY, self).__init__()
        
        # sample size checker, the first dimension is always the batch size
        assert X.shape[0] == Y.shape[0]
        
        self.X = X
        self.Y = Y

    # number of samples to be batched
    def __len__(self):
        return self.X.shape[0] 
       
    # get samples
    def __getitem__(self, index):
        return self.X[index], self.Y[index]
#-----------------------------------------------------------#



#---------------------------------------------------------------------------------------------#
# z-std method: apply componentwise z-standardization for feature scaling, e.g. see appendix 
class Zscaling:

    def __init__(self,):
        pass

    # Z standardized scaling
    # inputs:
    #       X: dataset, the first dim is always data size
    #                   the second dim is always feature size
    # Outputs:
    #      X: scaled dataset
    #      mu: mean vec
    #      std: standard deviation vec

    def z_std(self, X):
    
        # init mu and std vectors, 
        mu  = np.zeros(X.shape[1])
        std = np.zeros(X.shape[1])

        # taking mean and std at the example direction, one mu and one sigma per feature
        for i in range(X.shape[1]):

            mu[i]  = np.mean(X[:,i])
            std[i] = np.std(X[:,i])
        
        # normalize each feature by its own mean and std
        for i in range(X.shape[1]):    
            X[:,i] = (X[:,i] - mu[i])/std[i]        
        
        return X, mu, std

    # Z standardized scale-forward during validation stage
    # inputs:
    #       X: dataset to be scaled
    #       mu: mean from the entire dataset
    #       std: standard deviation from the entire dataset
    # Output:
    #       X: scaled dataset
    def scale_it_forward(self,X,mu,std):

        for i in range(X.shape[1]):
            
            X[:,i] = (X[:,i] - mu[i])/std[i]

        return X


    # Z standardized scale it back during validation stage
    # inputs:
    #       X: scaled dataset
    #       mu: mean from the entire dataset
    #       std: standard deviation from the entire dataset
    # output:
    #       X: dataset in its original scale
    def scale_it_back(self,X,mu,std):

        for i in range(X.shape[1]):
            
            X[:,i] = X[:,i] * std[i] + mu[i]

        return X

    # load scaling constants during validation and sampling stage
    # inputs:
        # folder_name: where to obtain those constants (file path)
    def load_scaling_constants(self, folder_name):
        muX = np.loadtxt(folder_name + '/muX.csv', delimiter=',', ndmin=1) 
        muY = np.loadtxt(folder_name + '/muY.csv', delimiter=',', ndmin=1)
        stdX = np.loadtxt(folder_name + '/stdX.csv', delimiter=',', ndmin=1)
        stdY = np.loadtxt(folder_name + '/stdY.csv', delimiter=',', ndmin=1)
        return muX, muY, stdX, stdY

    # Save scaling constants
    # Inputs:
    #        folder_name: where to save
    #        muX,muY, stdX, stdY: scaling constants from the entire dataset
    def save_scaling_constants(self, folder_name, muX, muY, stdX, stdY):
        np.savetxt(folder_name+'/muX.csv',  muX, delimiter = ',')
        np.savetxt(folder_name+'/muY.csv',  muY, delimiter = ',')
        np.savetxt(folder_name+'/stdX.csv', stdX, delimiter = ',')
        np.savetxt(folder_name+'/stdY.csv', stdY, delimiter = ',')
        return 0
#-----------------------------------------------------------------------------------------------------#



#-----------------------------------------------------------#
# training-testing dataset random split
# Input: 
#       samplesize: how many samples in total
#       T_portion: how much for gradient update
#       X: model input tensor
#       Y: model output tensor (to be learned), Y is usually set to None for training NF model solely
# Output:
#       sliced tensors
def TT_split(Sample_size, T_portion, X, Y):
    
    # how many training sample we have
    train_length = range(Sample_size)

    # pick random, un-ordered label from the range of all samples, non-repeat
    train_slice  = np.random.choice(train_length, size=int(T_portion*Sample_size), replace=False) 
    
    # do a set difference to get the test slice, this is random but ordered
    test_slice   = np.setdiff1d(train_length, train_slice) 
    
    #-------making slices out of random choices-------#
    # Note: we always use the 0-th dimension as the batch dimension

    # if train normalizing flow
    if Y == None:
        
        train_tensor        = X[train_slice,:]
        test_tensor         = X[test_slice,:]
        
        return train_tensor, test_tensor

    # if train emulator or variational decoder
    else:
        train_tensor        = X[train_slice,:]
        train_truth_tensor  = Y[train_slice,:]

        test_tensor        = X[test_slice,:]
        test_truth_tensor  = Y[test_slice,:]

        return train_tensor, train_truth_tensor, test_tensor, test_truth_tensor
#-----------------------------------------------------------#



#-----------------------------------------------------------#
# Nonlinear MLP function, can be generalized for various purposes
# inputs:         
        # NI: input size
        # NO: ouput size
        # NN: hidden size
        # NL: num of hidden layers
        # act: type of nonlinear activations, default: relu
# output:
#       sequential of layers

def MLP_nonlinear(NI,NO,NN,NL,act='relu'):

    # select act functions
    if act == "relu":
        actF = nn.ReLU()
    elif act == "tanh":
        actF = nn.Tanh()
    elif act == "sigmoid":
        actF = nn.Sigmoid()
    elif act == 'leaky':
        actF = nn.LeakyReLU(0.1)
    elif act == 'identity':
        actF = nn.Identity()
    elif act == 'silu':
        actF = nn.SiLU()

    #----------------construct layers----------------#
    MLP_layer = []

    # Input layer
    MLP_layer.append( nn.Linear(NI, NN) )
    MLP_layer.append(actF)
    
    # Hidden layer, if NL < 2 then no hidden layers
    for ly in range(NL-2):
        MLP_layer.append(nn.Linear(NN, NN))
        MLP_layer.append(actF)
   
    # Output layer
    MLP_layer.append(nn.Linear(NN, NO))
    
    # seq
    return nn.Sequential(*MLP_layer)
#-----------------------------------------------------------#



#-----------------------------------------------------------#
# general function to plot training and testing curves
# Inputs:
#        path: where to save the data
#        training: training losses
#        testing:  testing losses
#        args: plotting args

def TT_plot(PATH, training, testing, ylabel, yscale = 'normal' ):

    # plotting specs
    fs = 24
    plt.rc('font',  family='serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')
    plt.rc('text',  usetex=True)

    # plot loss curves
    fig1 = plt.figure(figsize=(10,8))

    # If apply axis scaling
    if yscale == 'semilogy':
        plt.semilogy(training, '-b', linewidth=2, label = 'Training');
        plt.semilogy(testing, '-r', linewidth=2, label = 'Testing');
    else:
        plt.plot(training, '-b', linewidth=2, label = 'Training');
        plt.plot(testing, '-r', linewidth=2, label = 'Testing');
    

    matplotlib.rc('font', size=fs+2)
    plt.xlabel(r'$\textrm{Epoch}$',fontsize=fs)
    plt.ylabel(ylabel,fontsize=fs)
    plt.tick_params(labelsize=fs+2)
    plt.legend(fontsize=fs-3)
       
    # save the fig   
    fig_name = PATH + '/'+ ylabel +'.png'
    plt.savefig(fig_name)


    # save the data
    train_name   = PATH + '/' + ylabel + '-train.csv'
    test_name    = PATH + '/' + ylabel + '-test.csv'
    
    np.savetxt(train_name, training,   delimiter = ',')
    np.savetxt(test_name, testing,   delimiter = ',')
            
    return 0
#----------------------------------------------------------------#


#----------------------------------------------------------------#
# Inner product test for the simple linear problem
# Inputs:
    # x: x samples from the inversion prediction
    # arg: which to check
# Outputs:
#  if fit line, return direction vector
#  if fit plane, return normal vector 
def line_plane_check(x, arg='line'):

    # normalization function
    def Normalize(v):
        return v/LA.norm(v,2)

    # exact kernel vector, un-normed
    kernel_vec             = np.array([1.0, -np.pi/np.exp(1.0), 1.0])
    
    # normalize the kernel vec
    normalized_kernel_vec  =  Normalize(kernel_vec)
    print('Exact kernel vector:' + str(normalized_kernel_vec))

    if arg == 'line': # do line check
        # apply line fit
        print('----------------------------')
        line_fit = Line.best_fit(x)
        print('line fit as:' + str(line_fit.direction))
        print('IP with kernel as:' + str(np.dot(  np.array(line_fit.direction) ,  normalized_kernel_vec)))
        print('----------------------------')
        return line_fit.direction
    elif arg == 'plane': # do plane check
        # apply plane fit
        print('----------------------------')
        plane_fit = Plane.best_fit(x)
        print('plane fit as:' + str(plane_fit.normal))
        print('IP with kernel as:' + str(np.dot(  np.array(plane_fit.normal) ,  normalized_kernel_vec)))
        print('----------------------------')
        return plane_fit.normal
#---------------------------------------------------------------------------#