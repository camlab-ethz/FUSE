# Normalizing flow via real NVP
# Code references:
#           ref: https://github.com/kamenbliznashki/normalizing_flows
#           ref: https://github.com/cedricwangyu/NoFAS
# Paper references:
#           ref: Variational inference with NoFAS: Normalizing flow with adaptive surrogate for computationally expensive models, Wang et.al
#           ref: Density estimation using Real NVP, Dinh et.al 2016
#           ref: Masked Autoregressive Flow for Density Estimation, Papamakarios et.al 2018

import torch
import torch.nn as nn
from DNN_tools import *
import torch.distributions as D


# Define NF class
class NF(nn.Module):
    def __init__(self, NF_in, NF_out, NF_hidden_unit, NF_hidden_layer, NF_block, act_s, act_t, BN ):
        super().__init__()

        # self parameters
        self.NF_in, self.NF_out = NF_in, NF_out    # input and output
        self.NF_hidden_unit     = NF_hidden_unit   # number of hidden unit
        self.NF_hidden_layer    = NF_hidden_layer  # number of hidden layer
        self.NF_block           = NF_block         # number of affine coupling blocks
        self.act_s              = act_s            # act for scaling function s
        self.act_t              = act_t            # act for translation function t
        self.BN                 = BN               # Boolean, if use batch norm
 

        # define base distribution to sample z, z~N(0,1)
        # Note: those paras will NOT be optimized during back-prop, but will be saved 
        #       in the ``state-dict"", i.e. it can be accessed later by the keywords
        self.register_buffer('Base_mean', torch.zeros(self.NF_in))
        self.register_buffer('Base_var', torch.ones(self.NF_in) )

        # constructing affine coupling layers 
        # Note: since we are not doing convolution, so we do not do masking here, otherwise
        #       inputs will contain padded zeros, making it less flexible
        # Note: we just follow what paper suggests: 1:d, d+1:D
        
        self.d = int( self.NF_in / 2 )

        # define slices 1:d and d+1:D 
        self.slice      = torch.arange(self.d)
        self.compliment = torch.arange(self.d, self.NF_in) 

        # start appending layers
        module_list = []

        # each block contains one coupling layer and a batch norm layer, if needed
        for nBlock in range(self.NF_block):

            module_list += [Affine_Coupling_Layers(self.slice, self.compliment, \
                                                    self.NF_hidden_unit, self.NF_hidden_layer,\
                                                        self.act_s, self.act_t)]

            # keep alternating structure by re-substitution
            temp            = self.slice
            self.slice      = self.compliment
            self.compliment = temp

            # if use batch normalization after each coupling layer
            if self.BN == True:
                module_list += self.BN * [Batch_Norm(self.NF_in) ]


        # define flows to forward/invert each module
        self.real_nvp_nf_net = Flow(*module_list)


    # forward: y --> z, turns z and Jacobian of the transformation
    def forward(self, y):
        return self.real_nvp_nf_net(y) # normalizing flow forward

    # inverse: z --> y
    def inverse(self, z):
        return self.real_nvp_nf_net.inverse(z) # normalizing flow backward, the sampler

    # define base distribution of z as standard normal 
    @property
    def base_dist(self):
        return D.Normal(self.Base_mean, self.Base_var)

    # compute log prob for training
    def LogProb(self, y):

        # we are using the following likelihood to train
        # log pY(y) = log pZ(z) + log|det(dZ dy)|

        # call the forward pass: y --> z
        z, sum_log_det_J = self.forward(y)

        # likelihood of z, the gaussian likelihood
        z_likelihood = torch.sum(self.base_dist.log_prob(z), dim=1).reshape((-1,1))

        return  z, z_likelihood + sum_log_det_J # c.f. equation 3 of Din

# batch norm of normalizing flows, applied to the end of each coupling layer
class Batch_Norm(nn.Module):

    def __init__(self, input_size, momentum = 0.9, eps = 1e-5):
        super().__init__()

        self.momentum  = momentum
        self.eps       = eps

        # addiditional trainables
        self.log_gamma = nn.Parameter(torch.zeros(input_size)) 
        self.beta      = nn.Parameter(torch.zeros(input_size))

        # register running statistics
        self.register_buffer('running_mean', torch.zeros(input_size))
        self.register_buffer('running_var', torch.ones(input_size))


    # forward process, batch norm w,r,t, inputs
    def forward(self, x):

        if self.training:

            # calculate batch statistics
            self.batch_mean = x.mean(0)
            self.batch_var  = x.var(0)

            # update running statistics, operation: from left to right
            self.running_mean.mul_(self.momentum).add_(self.batch_mean.data * (1.0 - self.momentum))
            self.running_var.mul_(self.momentum).add_(self.batch_var.data * (1.0 - self.momentum))

            # if during training, than use the batch stats
            mean = self.batch_mean
            var  = self.batch_var

        else:
            # if during testing or sampling, then use the running stats (accumulated)
            mean = self.running_mean
            var = self.running_var


        # batch normalization
        x_hat = (x - mean) / torch.sqrt( var + self.eps)
        
        # linear trans
        y     = self.log_gamma.exp() * x_hat + self.beta

        # compute Jacobian outof the transformation
        log_abs_det_jacobian = self.log_gamma - 0.5 * torch.log(var + self.eps)
        
        return y, torch.sum(log_abs_det_jacobian)*torch.ones(len(y),1)

    # backward process, batch norm w.r.t. predicted inputs
    def inverse(self, y):

        # running stats have been updated via the forward process
        if self.training:

            mean = self.batch_mean
            var  = self.batch_var
        else:
            mean = self.running_mean
            var  = self.running_var

        x_hat = (y - self.beta) * torch.exp(-self.log_gamma)
        
        x = x_hat * torch.sqrt(var + self.eps) + mean

        log_abs_det_jacobian = 0.5 * torch.log(var + self.eps) - self.log_gamma

        return x, torch.sum(log_abs_det_jacobian)*torch.ones(len(x),1)


#the affine coupling layers: ref: Dinh et al. 2017 equation 4,5,9
class Affine_Coupling_Layers(nn.Module):
    def __init__(self, Slice, Compliment, hidden_units, hidden_layer, act_s, act_t):
        super().__init__()

        self.slice      = Slice
        self.compliment = Compliment 
        self.hidden_units         = hidden_units
        self.hidden_layer         = hidden_layer
        self.act_s        = act_s
        self.act_t        = act_t

        # input-output of the scaling s and translation function t
        self.st_in, self.st_out = len(self.slice), len(self.compliment)

        # define scaling function ``s"" as plain MLP
        self.s = MLP_nonlinear(self.st_in, self.st_out, self.hidden_units, self.hidden_layer, act=self.act_s)
        
        # define translation function ``t"" as plain MLP
        self.t = MLP_nonlinear(self.st_in, self.st_out, self.hidden_units, self.hidden_layer, act= self.act_t)

    # forward function of the affine coupling layer
    # y -- > z
    # Note: the forward and inverse transformations defined here are different than
    #        either the reference code, this should be OK since we have a bijection here
    #        as long as we keep the correct log-likehood  
    def forward(self,y):

        # init latent var z
        z = torch.zeros_like(y)

        # apply slice to the input
        y_slice      = y[:,self.slice]
        y_compliment = y[:,self.compliment]

        # apply scaling function, computed here b/c Jacobian would need it as well 
        s_output = self.s(y_slice)

        # equ 7 of Dinh et .al
        z[:,self.slice]      = y_slice 
        z[:,self.compliment] = y_compliment * torch.exp( s_output ) + self.t(y_slice)

        # log det of Jacobian, c.f. Dinh et.al equation 6
        log_det_dzdy = torch.sum(s_output, dim = 1) # dim 1 is the feature dim

        return z, log_det_dzdy.reshape((-1,1)) 

    # from gaussian z back to y
    def inverse(self, z):

        # init sample y
        y = torch.zeros_like(z)

        # apply slice to the input
        z_slice      = z[:,self.slice]
        z_compliment = z[:,self.compliment]

        s_output = self.s(z_slice)

        # backward map
        y[:,self.slice]       =  z_slice
        y[:,self.compliment]  =  (z_compliment - self.t(z_slice)) * torch.exp(-s_output )

        # log det of jacobian
        log_det_dydz = -torch.sum(s_output, dim=1)

        return y, log_det_dydz.reshape( (-1,1))


# looping tho each component of the module list, gather jacobians
class Flow(nn.Sequential):
    
    # y --> z
    def forward(self, y):

        log_det_J_sum = 0 # start stacking Jacobian 

        for module in self: # looping tho each layer of the flow model

            z, log_det_J = module(y)   # forward pass

            log_det_J_sum += log_det_J # add the log jacobian of the current layer

            y = z # resubstitute

        return z, log_det_J_sum

    # z --> y
    def inverse(self, z):
        
        log_det_J_sum = 0 # start stacking Jacobian
        
        for module in reversed(self):

            y, log_det_J = module.inverse(z) # backward pass

            log_det_J_sum += log_det_J # add the log jacobian of the current layer

            z = y # resubstitute

        return y,log_det_J_sum