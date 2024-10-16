import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from lampe.inference import FMPE, FMPELoss


class FUSE(nn.Module):
    def __init__(self, fmpe_layers, fmpe_width, n_parameters, fno_in_layers, fno_out_layers, fno_width, fno_modes, channels_in, channels_target): 
        super(FUSE, self).__init__()
        # create two models: one to map the input (velocity) to parameters        
        self.FNO_in = FNO_in(fno_width, fno_modes, channels_in, fno_in_layers)
        self.FMPE_in = FMPE(n_parameters, 2*fno_modes*fno_width, hidden_features=[fmpe_width] * fmpe_layers)
        self.loss_in = FMPELoss(self.FMPE_in)
        # and one to map the parameters to the velocity
        self.fc0 = nn.Linear(n_parameters, fno_width) 
        self.fs0 = nn.Linear(1, 2*fno_modes) 
        self.FNO_out = FNO_out(fno_width, fno_modes, fno_width, channels_target, fno_out_layers)

    def param_loss(self, theta, x):
        # train the FMPE
        # theta: [batch size, number of parameters]
        # x:     [batch size, number of samples, number of points per sample]
        batch_size = theta.shape[0]
        
        # encode the data point to a vector
        x_spec = self.FNO_in(x)
        x_spec = x_spec.reshape(batch_size,-1)
        loss = self.loss_in(theta, x_spec)
        
        return loss
                                 
    def y_prediction(self, theta, n_points=181):
        # train the neural operator output from true parameters
        theta_enc = self.fc0(theta)[:,None,:]
        theta_enc = self.fs0(theta_enc.permute(0,2,1)).permute(0,2,1)

        y = self.FNO_out(theta_enc, n_points)
        return y
        
    def param_prediction(self, x, n_samples = 1000):  
        # predict the posterior distribution over the parameters
        # Compute encoded form of x
        x_spec = self.FNO_in(x)
        
        # Compute posterior distribution of theta given x
        batch_size = x.shape[0]
        theta = self.FMPE_in.flow(x_spec.reshape(batch_size,-1)).rsample((n_samples,))

        return theta

    def fused_loss(self, theta, n_points):
        batch_size = theta.shape[0]
        
        # train the neural operator output from true parameters
        theta_enc = self.fc0(theta)[:,None,:]
        theta_enc = self.fs0(theta_enc.permute(0,2,1)).permute(0,2,1)

        y = self.FNO_out(theta_enc, n_points)
        
        # encode the data point to a vector
        x_spec = self.FNO_in(y)
        x_spec = x_spec.reshape(batch_size,-1)
        loss = self.loss_in(theta, x_spec)
        return y, loss
                                 
    def full_flow(self, x, n_samples = 1000): 
        # predict the posterior distribution over the parameters 
        # and use these to compute the distribution over possible outputs
        n_points = x.shape[-1]
        # Compute encoded form of x
        x_spec = self.FNO_in(x)
        
        # Compute posterior distribution of theta given x
        batch_size = x.shape[0]
        theta = self.FMPE_in.flow(x_spec.reshape(batch_size,-1)).rsample((n_samples,))
        
        # Compute encoded form of theta
        theta_enc = self.fc0(theta)
        theta_enc = self.fs0(theta_enc.permute(0,2,1)).permute(0,2,1)

        # calculate possible outputs
        y = self.FNO_out(theta_enc, n_points)

        return theta, y
   
    
# class for fully nonequispaced 1d points
class VFT:
    def __init__(self, batch_size, n_points, modes):
                
        self.number_points = n_points
        self.batch_size = batch_size
        self.modes = modes

        new_times = (torch.arange(self.number_points)/self.number_points).repeat(self.batch_size, 1).cuda()
        
        self.new_times = new_times * 2 * np.pi
        
        self.X_ = torch.arange(modes).repeat(self.batch_size, 1)[:,:,None].float().cuda()
        self.V_fwd, self.V_inv = self.make_matrix()

    def make_matrix(self):
        X_mat = torch.bmm(self.X_, self.new_times[:,None,:])
        forward_mat = torch.exp(-1j* (X_mat)) 

        inverse_mat = torch.conj(forward_mat.clone()).permute(0,2,1) 

        return forward_mat, inverse_mat

    def forward(self, data, norm='forward'):
        data_fwd = torch.bmm(self.V_fwd, data)
        if norm == 'forward':
            data_fwd /= self.number_points
        elif norm == 'ortho':
            data_fwd /= np.sqrt(self.number_points)
            
        return data_fwd

    def inverse(self, data, norm='backward'):
        data_inv = torch.bmm(self.V_inv, data) 
        if norm == 'backward':
            data_inv /= self.number_points
        elif norm == 'ortho':
            data_inv /= np.sqrt(self.number_points)
            
        return data_inv
    

class SpectralConv1d_SMM (nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super(SpectralConv1d_SMM, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x, transform):
        batchsize = x.shape[0]

        x = x.permute(0, 2, 1)

        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = transform.forward(x.cfloat(), norm='backward')
        x_ft = x_ft.permute(0, 2, 1)

        # # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, self.modes, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes] = self.compl_mul1d(x_ft, self.weights1)

        # #Return to physical space

        x_ft = out_ft.permute(0, 2, 1)
        x = transform.inverse(x_ft, norm='backward') # x [4, 20, 512, 512]
        x = x.permute(0, 2, 1)

        return x.real
    
    def last_layer(self, x, transform):
        x = x.permute(0, 2, 1)

        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = transform.forward(x.cfloat(), norm='forward') 
        
        x_ft = x_ft.permute(0, 2, 1)
        x_ft = self.compl_mul1d(x_ft, self.weights1)
        
        
        return torch.view_as_real(x_ft).reshape(x.shape[0], x.shape[2], -1)
    
    def first_layer(self, x, transform):
        
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2]//2, 2).contiguous()
        x_ft = torch.view_as_complex(x)
        
        x_ft = self.compl_mul1d(x_ft, self.weights1)
        
        x_ft = x_ft.permute(0, 2, 1)
        x = transform.inverse(x_ft, norm='forward') # x [4, 20, 512, 512]
        x = x.permute(0, 2, 1)
        
        
        return x.real



class FNO_in (nn.Module):
    def __init__(self, width, modes, channels_in, layers):
        super(FNO_in, self).__init__()

        self.modes = modes
        self.width = width

        # Define network
        self.fc0 = nn.Linear(channels_in, self.width) 
        
        self.conv_layers = nn.ModuleList([
            # SpectralConv1d(self.width, self.width, self.modes) for _ in range(layers)
            SpectralConv1d_SMM(self.width, self.width, self.modes) for _ in range(layers)
        ])
        self.w_layers = nn.ModuleList([
            nn.Conv1d(self.width, self.width, 1) for _ in range(layers)
        ])
        
        self.conv_last = SpectralConv1d_SMM(self.width, self.width, self.modes)

    def forward(self, x):
        batch_size = x.shape[0]
        n_points = x.shape[-1]
        # dont need to concatenate a grid, because this is already provided (as time) in the data
        x = x.permute(0, 2, 1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        
        
        fourier_transform = VFT(batch_size, n_points, self.modes)
        
        for conv, w in zip(self.conv_layers, self.w_layers):
            x1 = conv(x, fourier_transform)
            x2 = w(x)
            x = x1 + x2
            x = F.gelu(x)
        
        # if running with FNO spectral output
        x_spec = self.conv_last.last_layer(x, fourier_transform)
        x_spec = x_spec.permute(0, 2, 1)
        
        return x_spec
    
class FNO_out (nn.Module):
    def __init__(self, width, modes, channels_in, channels_target, layers):
        super(FNO_out, self).__init__()

        self.modes = modes
        self.width = width
        

        # Define network
        self.fc0 = nn.Linear(channels_in, self.width) 
        
        self.conv_layers = nn.ModuleList([
            # SpectralConv1d(self.width, self.width, self.modes) for _ in range(layers)
            SpectralConv1d_SMM(self.width, self.width, self.modes) for _ in range(layers)
        ])
        self.w_layers = nn.ModuleList([
            nn.Conv1d(self.width, self.width, 1) for _ in range(layers)
        ])

        self.conv_first = SpectralConv1d_SMM(self.width, self.width, self.modes)
        
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, channels_target)

    def forward(self, x_spec, n_points):
        # x is shape [batch, points, channels]
        batch_size = x_spec.shape[0]
        x_spec = self.fc0(x_spec)
        x_spec = x_spec.permute(0, 2, 1)
        
        fourier_transform = VFT(batch_size, n_points, self.modes)
        x = self.conv_first.first_layer(x_spec, fourier_transform)
        
        for conv, w in zip(self.conv_layers, self.w_layers):
            # x1 = conv(x)
            x1 = conv(x, fourier_transform)
            x2 = w(x)
            x = x1 + x2
            x = F.gelu(x)
            
        x = x1.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        
        x = x.permute(0, 2, 1)
        return x
    
