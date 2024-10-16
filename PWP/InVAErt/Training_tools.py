import os
from Model import *
from DNN_tools import *
import math
import torch
import torch.nn as nn
import pdb

import sys
sys.path.append('../')
from utilities import *

#-----------------------------------------------------------------------------------------#
# setup trainer class
class Training:
    def __init__(self, X, Y, para_dim, train_tensor=None, \
                                        train_truth_tensor=None,\
                                         test_tensor=None,\
                                          test_truth_tensor=None, latent_plus = 0):
        
        self.X = X               # input dataset: num of data \times input feature
        self.Y = Y               # input dataset: num of data \times output feature
        self.para_dim = para_dim # dim(V), we need to exclude the size of auxillary data

        #--------------------------------------find NN parameters------------------------------------------------#
        # forward encoder parameter (emulator of the forward process)
        self.input_encoder  = self.X.shape[1] # dim(V) + dim(D_v)
        self.output_encoder = self.Y.shape[1] # dim(Y)

        # NF sampler parameter (sampler of model output)
        self.input_NF       = self.output_encoder # dim(Y)
        self.output_NF      = self.output_encoder # dim(Y)

        # VAE sampler parameter (non-id manifold)
        self.input_VAE_encoder      = self.para_dim # VAE encoder input, note that dim(D_v) is removed


        # Allowing higher dimension representation of the latent space, e.g. see Section 2.3
        #         latent_plus: additional dimension to be added for W, default is 0
        #---------------------------------------------------------------------------------#
        self.Latent_VAE             = self.para_dim - self.output_encoder + latent_plus # number of latent dimension, i.e. w's dimension, dim(W)
        assert self.Latent_VAE > 0, "need a dimensional reduction problem" # general extension TBD
        #---------------------------------------------------------------------------------#
        
        self.output_VAE_encoder     = 2*self.Latent_VAE # mean and variance of the latent var w, combined, i.e. [mu1, mu2, ..., std1, std2, ...]

        # decoder parameter (the inverse problem)
        self.input_decoder  = self.output_encoder + self.Latent_VAE # at least creating a bijection
        self.output_decoder = self.para_dim                         # dim(V)
        #---------------------------------------------------------------------------------------------------------#

        #---------------------self dataset for later training---------------------------#
        self.train_T, self.train_truth_T, self.test_T, self.test_truth_T = \
                        train_tensor, train_truth_tensor, test_tensor, test_truth_tensor
        #-------------------------------------------------------------------------------#
    

    #-----------------------------------------------------------------------------------------#
    # define each component of inVAErt network
    # Inputs: 
    #      device: cpu or gpu
    #      encoder_para: encoder NN parameters: hidden unit, hidden layer, activation function
    #      nf_para:      real-nvp parameters:   hidden unit, hidden layer, affine coupling blocks, if using batch norm
    #      vae_para: hidden unit, hidden layer, activation function
    #      decoder_para: hidden unit, hidden layer, activation function
    # Note: decoder and vae are connected
    def Define_Models(self, device, encoder_para, nf_para, vae_para, decoder_para):

        #--------------------------------Forward encoder def (emulator)-------------------------------------#
        model_encoder = Encoder(self.input_encoder, self.output_encoder, encoder_para[0], encoder_para[1], encoder_para[2])
        
        self.model_encoder = model_encoder.to(device) 

        #print(self.model_encoder)
        Encoder_params = sum(p.numel() for p in self.model_encoder.parameters() if p.requires_grad)
        print( 'Number of trainable para for encoder is:'  + str(Encoder_params) )
        #-----------------------------------------------------------------------------------------#


        #--------------------------------NF Real-NVP def---------------------------------------------------#
        if self.input_NF == 1:
            print('y is a scalar, adding one aux dimension...')
            model_NF = NF_sampler(self.input_NF+1, self.output_NF+1, nf_para[0], nf_para[1], nf_para[2], nf_para[3]) 
        else:
            model_NF = NF_sampler(self.input_NF, self.output_NF, nf_para[0], nf_para[1], nf_para[2], nf_para[3])
        
        self.model_NF = model_NF.to(device) 

        # print(self.model_NF)
        NF_params = sum(p.numel() for p in self.model_NF.parameters() if p.requires_grad)
        print( 'Number of trainable para for NF sampler is:'  + str(NF_params) )
        #--------------------------------------------------------------------------------------------------#

        #-------------------------------VAE+Decoder def------------------------------------------#
        model_decoder = VAEDecoder( self.input_VAE_encoder, self.output_VAE_encoder, self.input_decoder, \
                                        self.output_decoder, vae_para, decoder_para  )
        
        self.model_decoder = model_decoder.to(device) 

        #print(self.model_decoder)
        Decoder_params = sum(p.numel() for p in self.model_decoder.parameters() if p.requires_grad)
        print( 'Number of trainable para for VAE+decoder is:'  + str(Decoder_params) )
        #-----------------------------------------------------------------------------------------#

        
        return self.model_encoder, self.model_NF, self.model_decoder
    #-------------------------------------------------------------------------------------------------------#
    

    #-------------------------------------------------------------------------------------------------------#
    # Training and testing steps for the emulator N_e
    # inputs:
    #        PATH: where to save the model
    #        model: previously defined encoder model, to be trained
    #        lr_min: minimum learning rate
    #        lr: initial learning rate
    #        decay: learning rate decay rate
    #        nB: mini-batch size
    #        l2_decay: penalty for l2 decay, default=0
    #        residual: if apply ResNet, default: False
    def Encoder_train_test(self, PATH, model, lr_min, lr, decay, nB, l2_decay=0, residual = False):
        print('\n')
        print('---------------Start to train the encoder-----------------')
        
        # calculate total number of epoches
        num_epochs     = 2000 #int(math.log(lr_min/lr, decay))

        print('Total number of epoches for training encoder is:' + str(num_epochs))

        # define optimizer, always use Adam first
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_decay)
        
        # build lr scheduler
        lambda1 = lambda epoch: decay ** epoch # lr scheduler by lambda function
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda1) # define lr scheduler with the optim

        # init loss/acc for encoder
        Emeasure = nn.MSELoss() # mse for error measure
        train_save, train_acc_save = [],[]
        test_save,  test_acc_save  = [],[]

        #-----------------------Start-training-testing-------------------------#
        for epoch in range(num_epochs):

            # for each epoch, save the loss for plotting 
            train_loss_per, train_acc_per = 0,0
            test_loss_per,  test_acc_per  = 0,0

            #----------------------use dataloader to do auto-batching--------------------------------#
            traindata    =  MyDatasetXY(self.train_T, self.train_truth_T)
            trainloader  =  torch.utils.data.DataLoader(traindata, batch_size=nB, shuffle=True) # always shuffle the training batches after each epoch
    
            testdata     =  MyDatasetXY(self.test_T, self.test_truth_T)
            testloader   =  torch.utils.data.DataLoader(testdata, batch_size=nB, shuffle=False) # no need to shuffle the testing batches
            
            num_batches_train = len(trainloader) # total number of training mini batches
            num_batches_test  = len(testloader)  # total number of testing mini batches
            
            if epoch == 0:
                print('Encoder: Total num of training batches:' + str(num_batches_train)+ ', testing batches:' + str(num_batches_test))
            #--------------------------------------------------------------------#

            # start training
            model.train()

            # ------------------------Training: loop tho minibatches----------------------------#
            for X,Y in trainloader:

                # model forward 
                Y_hat = model(X, residual=residual, res_num=Y.shape[1]) # if apply residual, residual feature dimension is alawys y's feature dimension, stacked at the end of "X"

                # loss function via mse functional
                Loss     = Emeasure(Y_hat, Y)
                
                # MSE accuracy
                Accuracy = (  1.0 - Loss  / Emeasure( Y, torch.zeros_like(Y) )    )*100
                
                # keep track of the loss
                train_loss_per += Loss
                train_acc_per  += Accuracy

                # Zero-out the gradient
                opt.zero_grad()

                # back-prop
                Loss.backward()

                # gradient update
                opt.step()
            # ---------------------------------------------------------------------------------- #
            
            # ---------------------------save epoch-wise quantities----------------------------- #
            train_save.append(train_loss_per.item()/num_batches_train) # per batch loss
            train_acc_save.append(train_acc_per.item()/num_batches_train) # per batch accuracy
            
            # for every xx epoches, print statistics for monitering
            if epoch%10 == 0:
                print("Training: Epoch: %d, forward loss: %1.3e" % (epoch, train_save[epoch]) ,\
                    ", forward acc : %.6f%%"  % (train_acc_save[epoch]), \
                    ', current lr: %1.3e' % (opt.param_groups[0]['lr']))

            # update learning rate via scheduler
            scheduler.step()    
            #------------------------------------------------------------------------------------#


            # start testing
            model.eval()
            # -------------------------Testing: loop tho minibatches-----------------------------#
            l1_errors = torch.zeros(128)
            l2_errors = torch.zeros(128)
            iter = 0
            with torch.no_grad():
                for X,Y in testloader:

                    # loss function via mse functional
                    Loss_test     = Emeasure(model(X, residual=residual, res_num=Y.shape[1]), Y)
                
                    # MSE accuracy
                    Accuracy_test = (  1.0 - Loss_test / Emeasure( Y, torch.zeros_like(Y) )   )*100

                    test_loss_per += Loss_test
                    test_acc_per  += Accuracy_test
                    
                    pressure_pred = model(X, residual=residual, res_num=Y.shape[1]).reshape(X.shape[0], 52, 487)
                    pressure_pred = pressure_pred[:,:13,:]
                    
                    pressure_true = Y.reshape(X.shape[0], 52, 487)[:,:13,:]
                    
                    
                    l1_errors[iter*nB:(iter+1)*nB] = torch.mean(relative_error(pressure_pred, pressure_true, p=1), 1)
                    l2_errors[iter*nB:(iter+1)*nB] = torch.mean(relative_error(pressure_pred, pressure_true, p=2), 1)
                    
                    iter += 1
            # -----------------------------------------------------------------------------------#

            # ---------------------------save epoch-wise quantities----------------------------- #
            test_save.append(test_loss_per.item()/num_batches_test)
            test_acc_save.append(test_acc_per.item()/num_batches_test)
            
            # for every xx epoches, print statistics for monitering
            if epoch%200 == 0:
                print("Testing: forward loss: %1.3e" % (test_save[epoch]) ,\
                    ", forward acc : %.6f%%"  % (test_acc_save[epoch]))
                print(f"l1 error: {torch.mean(l1_errors).item():.4f} +/- {torch.std(l1_errors).item():.4f}")
                print(f"l2 error: {torch.mean(l2_errors).item():.4f} +/- {torch.std(l2_errors).item():.4f}")
            # -----------------------------------------------------------------------------------  #

            # -------------------- invoke early stop to end training if nec----------------------------- #
            # TBD
            #---------------------------------------------------------------------------------------------#

            #------------------------------save the model--------------------------------------#
            if epoch % 200 == 0 or epoch == num_epochs-1: # for in-time evaluation, if needed
                print('Saving results....')
                # save trained weights
                model_save_name   = PATH + '/Encoder_model.pt'
                torch.save(model, model_save_name)
                # plot training/testing loss curves
                # TT_plot(PATH, train_save, test_save, 'EncoderLoss', yscale = 'semilogy' )
                # TT_plot(PATH, train_acc_save, test_acc_save, 'EncoderAccuracy')
            #-----------------------------------------------------------------------------------#

        return None
        #-----------------------------------------------------------------------------------------#



    #-----------------------------------------------------------------------------------------#
    # apply training and testing for the Real-NVP based normalizing flow model
    # inputs:
    #        PATH: where to save the model
    #        model: NF real NVP model, to be trained
    #        lr_min: minimum learning rate
    #        lr: initial learning rate
    #        decay: learning rate decay rate
    #        nB: mini-batch size
    #        l2_decay: penalty for l2 decay, default=0
    def NF_train_test(self, PATH, model, lr_min, lr, decay, nB, l2_decay=0):
        
        print('\n')
        print('---------------Start to train the Real NVP-----------------')
        
        #--------------------------------------------------------------------#
        # calculate number of epoches
        num_epochs     = 2000 #int(math.log(lr_min/lr, decay))

        print('Total number of epoches for training NF is:' + str(num_epochs))

        # define optimizer, always use Adam first
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_decay)
        
        # build lr scheduler
        lambda1 = lambda epoch: decay ** epoch # lr scheduler by lambda function
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda1) # define lr scheduler with the optim

        # init likelihood loss for nf
        train_NF_save = []
        test_NF_save  = []
        #----------------------------------------------------------------------#


        #-----------------------Start-training-testing-------------------------#
        for epoch in range(num_epochs):

            # for each epoch, save the loss for plotting 
            train_NF_per, test_NF_per = 0,0

            #----------------------use dataloader to do auto-batching--------------------------------#
            # Note: here we feed dataloader with the truth, i.e. y's
            traindata    =  MyDatasetX(self.train_truth_T)
            trainloader  =  torch.utils.data.DataLoader(traindata, batch_size=nB, shuffle=True) # always shuffle the training batches after each epoch
    
            testdata     =  MyDatasetX(self.test_truth_T)
            testloader   =  torch.utils.data.DataLoader(testdata, batch_size=nB, shuffle=False) # no need to shuffle the testing batches
            
            num_batches_train = len(trainloader) # total number of training mini batches
            num_batches_test  = len(testloader)  # total number of testing mini batches
            
            if epoch == 0:
                print('NF: Total num of training batches:' + str(num_batches_train)+ ', testing batches:' + str(num_batches_test))
            #--------------------------------------------------------------------#

            # start training
            model.train()

            # ------------------------Training: loop tho minibatches----------------------------#
            for Y in trainloader:

                # if y is a scalar, adding one aux standard gaussian variable
                if Y.shape[1] == 1:
                    Y = torch.cat( (Y, torch.normal(0,1, size=(Y.shape[0],1))), dim = 1 )

                # forward transformation and get the log likelihood
                Z, MiniBatchLogLikelihood = model(Y)

                Loss           = -1 * torch.mean(MiniBatchLogLikelihood, dim = 0) # take mean w.r.t mini-batches
                train_NF_per  += Loss

                # Zero-out the gradient
                opt.zero_grad()

                # back-prop
                Loss.backward()

                # gradient update
                opt.step()
            #--------------------------------------------------------------------------------------#

            # ---------------------------save epoch-wise quantities----------------------------- #
            train_NF_save.append(train_NF_per.item()/num_batches_train)
            
            # for every xx epoches, print statistics for monitering
            if epoch%200 == 0:
                print("Training: Epoch: %d, likelihood loss: %1.3e" % (epoch, train_NF_save[epoch]) ,\
                    ', current lr: %1.3e' % (opt.param_groups[0]['lr']))

            # update learning rate via scheduler
            scheduler.step()    
            #--------------------------------------------------------------------------------------#
            
            # start testing
            model.eval()
            # -------------------------Testing: loop tho minibatches-------------------------------#
            with torch.no_grad():
                for Y in testloader:

                    # if y is a scalar, adding one aux standard gaussian variable
                    if Y.shape[1] == 1:
                        Y = torch.cat( (Y, torch.normal(0,1, size=(Y.shape[0],1))), dim = 1 )

                    # forward transformation and get the log likelihood
                    Z, MiniBatchLogLikelihood = model(Y)

                    #--------check if NF is correct-----------#
                    #Y_hat = model.inverse(Z)
                    #print(torch.norm(Y-Y_hat))
                    #-----------------------------------------#

                    Loss          = -1 * torch.mean(MiniBatchLogLikelihood, dim = 0) # take mean w.r.t mini-batches
                    test_NF_per  += Loss
            
            # ---------------------------save epoch-wise quantities----------------------------- #
            test_NF_save.append(test_NF_per.item()/num_batches_test)

            # for every xx epoches, print statistics for monitering
            if epoch%10 == 0:
                print("Testing: likelihood loss: %1.3e" % (test_NF_save[epoch]))
            #--------------------------------------------------------------------------------------#

            # -------------------- invoke early stop to end training if nec----------------------------- #
            # TBD
            #---------------------------------------------------------------------------------------------#

            #------------------------------------save the model-----------------------------------------#
            if epoch % 1 == 0 or epoch == num_epochs-1:
                # save trained weights
                model_save_name   = PATH + '/NF_model.pt'
                torch.save(model, model_save_name)
                # # plot likelihood loss curves
                # TT_plot(PATH, train_NF_save, test_NF_save, 'LogNFLikelihood')
            #---------------------------------------------------------------------------------------------#

        return None
        #-----------------------------------------------------------------------------------------#
    


    #-----------------------------------------------------------------------------------------#
    # apply training and testing to variational decoder (i.e. VAE + decoder)
    # inputs:
    #        PATH: where to save the model
    #        model: variational decoder model, to be trained (i.e. VAE + decoder)
    #        lr_min: minimum learning rate
    #        lr: initial learning rate
    #        decay: learning rate decay rate
    #        nB: batch size
    #        penalty: importance assigned to kl divergence, decoder mse loss and encoder reconstraint loss (not always needed)
    #        l2_decay: l2 decay factor
    #        EN: trained emulator for imposing the reconstraint loss Lr (default: None, i.e. not used)
    #        residual: if apply residual to the emulator (if EN is True to impose Lr), default: True
    
    # Note: if the trained Emulator is used for imposing the reconstraint loss Lr, we just forward it with each mini-batch 
    #       without fine-tuning it, since tuning will make it worse b/c the variational decoder is hard to train.
    def Decoder_train_test(self, PATH, model, lr_min, lr, decay, nB, penalty, \
                                         l2_decay=0, EN = None, residual = True):
        print('\n')
        print('---------------Start to train the variational decoder-----------------')
        
        # calculate number of epoches
        num_epochs     = int(math.log(lr_min/lr, decay))

        print('Total number of epoches for training variational decoder is:' + str(num_epochs))

        # define optimizer, always try Adam first
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_decay)
            
        # build lr scheduler
        lambda1 = lambda epoch: decay ** epoch # lr scheduler by lambda function
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda1) # define lr scheduler with the optim
        
        # init loss/acc for VAE and decoder
        Emeasure = nn.MSELoss() # mse for error measure
        train_save, train_acc_save = [],[] # decoder training loss and accuracy
        test_save,  test_acc_save  = [],[] # decoder testing loss and accuracy
        train_KL_save, test_KL_save = [], [] # KL divergence loss: training and testing
        train_enc_save, test_enc_save = [], [] # emulator reconstraint loss and accuracy, if needed

        #-----------------------Start-training-testing-------------------------#
        for epoch in range(num_epochs):

            # for each epoch, save the loss for plotting 
            train_loss_per, train_acc_per = 0,0
            test_loss_per,  test_acc_per  = 0,0
            train_KL_per,   test_KL_per   = 0,0

            # if the trained emulator is used for enforcing re-constraint loss, then init loss per epoch
            #===========================================================#
            if EN != None:
                train_encoder_again, test_encoder_again = 0, 0 
            #===========================================================#
            
            #----------------------use dataloader to do auto-batching--------------------------------#
            traindata    =  MyDatasetXY(self.train_T, self.train_truth_T)
            trainloader  =  torch.utils.data.DataLoader(traindata, batch_size=nB, shuffle=True)
    
            testdata     =  MyDatasetXY(self.test_T, self.test_truth_T)
            testloader   =  torch.utils.data.DataLoader(testdata, batch_size=nB, shuffle=False)
            
            num_batches_train = len(trainloader) # total number of training mini batches
            num_batches_test  = len(testloader)  # total number of testing mini batches
            
            if epoch == 0:
                print('VAEDecoder: Total num of training batches:' + str(num_batches_train)+ ', testing batches:' + str(num_batches_test))
            #--------------------------------------------------------------------#

            # start training
            model.train() # training mode for VAE and decoder

            #===========================================================#
            if EN != None: # if use Lr, the reconstraint loss
                EN.eval()  # never fine-tune the trained emulator, put trained emulator in eval mode
            #===========================================================#
            
            # ------------------------Training: loop tho minibatches----------------------------#
            for X,Y in trainloader:
                #-do not use aux data in VAE encoding-#
                X_aux = X[:,self.para_dim:] # this is the aux data
                X     = X[:,:self.para_dim] # this is the v
                #-------------------------------------#
                
                
                Y_ = Y.reshape(Y.shape[0], 52, 487)
                
                channels = Y_.shape[1]
                mask = random_channel_mask(Y_.shape[0], input_channels=channels).cuda()
                Y_ = Y_ * mask.view(Y_.shape[0], channels, 1)
                
                Y = Y_.reshape(Y.shape[0], 52*487)
                
                # model forward, note that kl_loss is already the mean w.r.t the current mini-batch
                kl_loss, X_hat = model(X, Y)
                

                # loss function via mse functional, i.e. re-construction loss
                Loss     = Emeasure(X_hat, X) 
                
                # MSE inversion accuracy
                Accuracy = (  1.0 - Loss  / Emeasure( X, torch.zeros_like(X) )    )*100

                #==========================================================================================#
                # if imposing the re-constraint loss Lr, this is so called the knowledge distill
                if EN != None:
                    with torch.no_grad(): # no gradient calculation for EN since it is already trained
                        
                        # cat prediction and true aux data
                        X_hat_encoding          = torch.cat(( X_hat, X_aux  ),dim=1)
                        # forward the trained emulator
                        Y_hat                   = EN(X_hat_encoding, residual=residual, res_num=Y.shape[1])
                        # re-evaluate the mse forward loss
                        Loss_encoding_again     = Emeasure(Y_hat, Y)
                        # keep track of the loss
                        train_encoder_again += Loss_encoding_again
                #==========================================================================================#

                # keep track of the losses
                train_loss_per += Loss      # decoder re-construction loss
                train_acc_per  += Accuracy  # decoder re-construction accuracy
                train_KL_per   += kl_loss   # KL loss

                # Zero-out the gradient
                opt.zero_grad()

                # back-prop
                if EN == None: # if no re-constraint loss Lr
                    (penalty[0]*kl_loss + penalty[1]*Loss).backward()
                else:          # if use re-constraint loss Lr
                    (penalty[0]*kl_loss + penalty[1]*Loss + penalty[2]*Loss_encoding_again).backward()
                
                # gradient update
                opt.step()
            # ---------------------------------------------------------------------------------- #

            # ---------------------------save and plot for training----------------------------- #
            train_save.append(train_loss_per.item()/num_batches_train)
            train_KL_save.append(train_KL_per.item()/num_batches_train)
            train_acc_save.append(train_acc_per.item()/num_batches_train)
            
            #===========================================#
            if EN != None: # if use Lr
                train_enc_save.append(train_encoder_again.item()/num_batches_train)
            #===========================================#

            if epoch%200 == 0:
                #===========================================#
                if EN != None:
                    print("Training: Epoch: %d, inversion loss: %1.3e" % (epoch, train_save[epoch]) ,\
                        ", inversion acc : %.6f%%"  % (train_acc_save[epoch]), \
                        ", KL : %1.3e"  % (train_KL_save[epoch]), \
                        ", Encoder_reconstraint: %1.3e" % (train_enc_save[epoch]), \
                        ', current lr: %1.3e' % (opt.param_groups[0]['lr']))
                #===========================================#
                else:
                    print("Training: Epoch: %d, inversion loss: %1.3e" % (epoch, train_save[epoch]) ,\
                        ", inversion acc : %.6f%%"  % (train_acc_save[epoch]), \
                        ", KL : %1.3e"  % (train_KL_save[epoch]), \
                        ', current lr: %1.3e' % (opt.param_groups[0]['lr']))

            # update learning rate via scheduler
            scheduler.step()    
            #------------------------------------------------------------------------------------#

            # start testing
            model.eval()
            # -------------------------Testing: loop tho minibatches-----------------------------#
            with torch.no_grad():
                for X,Y in testloader:

                    Y_ = Y.reshape(Y.shape[0], 52, 487)

                    channels = Y_.shape[1]
                    mask = random_channel_mask(Y_.shape[0], input_channels=channels).cuda()
                    Y_ = Y_ * mask.view(Y_.shape[0], channels, 1)

                    Y = Y_.reshape(Y.shape[0], 52*487)
                    
                    #-do not use aux data in VAE encoding-#
                    X_aux = X[:,self.para_dim:]  # this is the aux data
                    X     = X[:,:self.para_dim]  # this is the v
                    #-------------------------------------#

                    # model forward, note that kl_loss is already the mean w.r.t the current mini-batch
                    kl_loss, X_hat = model(X, Y)

                    # loss function via mse functional, i.e. re-construction loss
                    Loss     = Emeasure(X_hat, X) 
                    
                    # MSE inversion accuracy
                    Accuracy = (  1.0 - Loss  / Emeasure( X, torch.zeros_like(X) )    )*100

                    #=====================================================================================#
                    # re-evaluate emulator if needed (already torch.no_grad())
                    if EN != None:

                        # cat prediction and true aux data
                        X_hat_encoding          = torch.cat(( X_hat, X_aux  ), dim=1)
                        # forward the trained emulator
                        Y_hat                   = EN(X_hat_encoding, residual=residual, res_num=Y.shape[1])
                        # re-evaluate the mse forward loss
                        Loss_encoding_again     = Emeasure(Y_hat, Y)
                        # keep track of the loss
                        test_encoder_again      += Loss_encoding_again
                    #=======================================================================================#

                    # keep track of the losses
                    test_loss_per += Loss
                    test_acc_per  += Accuracy
                    test_KL_per   += kl_loss
            #------------------------------------------------------------------------------------#
             
             #---------------------------save and plot for training----------------------------- #
            test_save.append(test_loss_per.item()/num_batches_test)
            test_KL_save.append(test_KL_per.item()/num_batches_test)
            test_acc_save.append(test_acc_per.item()/num_batches_test)

            #===========================================#
            if EN != None:  # if use Lr
                test_enc_save.append(test_encoder_again.item()/num_batches_test)
            #===========================================#
            
            if epoch%20 == 0:
                #===========================================#
                if EN != None:
                    print("Testing: inversion loss: %1.3e" % (test_save[epoch]) ,\
                        ", inversion acc : %.6f%%"  % (test_acc_save[epoch]), \
                        ", KL : %1.3e"  % (test_KL_save[epoch]),\
                        ", Encoder_reconstraint: %1.3e" % (test_enc_save[epoch]))
                #===========================================#

                else:
                    print("Testing: inversion loss: %1.3e" % (test_save[epoch]) ,\
                        ", inversion acc : %.6f%%"  % (test_acc_save[epoch]), \
                        ", KL : %1.3e"  % (test_KL_save[epoch]))
            # -----------------------------------------------------------------------------------  #

            # -------------------- invoke early stop to end training if nec----------------------------- #
            # TBD
            #---------------------------------------------------------------------------------------------#

            #---------------------------save the model-------------------------------#
            if epoch % 20 == 0 or epoch == num_epochs-1:
                
                # save the trained weights
                model_save_name   = PATH + '/Decoder_model.pt'
                torch.save(model, model_save_name)
                
#                 # plot loss curves
#                 TT_plot(PATH, train_save, test_save, 'DecoderLoss', yscale = 'semilogy' )
#                 TT_plot(PATH, train_acc_save, test_acc_save, 'DecoderAccuracy')
#                 TT_plot(PATH, train_KL_save, test_KL_save, 'KL divergence')
                
#                 #===========================================#
#                 if EN != None:
#                     TT_plot(PATH, train_enc_save, test_enc_save, 'Encoder_reconstraint', yscale = 'semilogy')
#                 #===========================================#
            #-------------------------------------------------------------------------#

        return None
        #-----------------------------------------------------------------------------------------#


#----------------------------------------------------------------#
# load trained model
# Inputs:
#       device: cpu or gpu
#       folder_name: where to find trained models
#       X,Y: training dataset, to obtain dimension information
#       para_dim: number of model input parameters, excluding the aux data dimension
#       encoder_paras, nf_paras, var_paras, decoder_paras: hidden/ layer/ act/ BN info for models
#       latent_plus: if needed, expand latent dimensions
# Outputs: 
#      Encoder, NF, Decoder: trained model loaded
def model_loading(device, folder_name, X,Y, para_dim, encoder_para, nf_para, vae_para, decoder_para, latent_plus = 0):
    
    #-----------------------------load model------------------------------------------------#
    Trainer = Training(X,Y,para_dim,latent_plus = latent_plus) # use training data to get the dimension information only
    Encoder, NF, Decoder = Trainer.Define_Models(device, encoder_para, nf_para, vae_para, decoder_para)

    # load trained weights for encoder
    encoder_path = folder_name + '/Encoder_model.pt'
    if os.path.exists(encoder_path):
        Encoder.load_state_dict(torch.load(encoder_path, map_location=device)) # load learned weights
    else:
        pass

    # load trained weights for NF
    nf_path = folder_name + '/NF_model.pt'
    if os.path.exists(nf_path):
        NF.load_state_dict(torch.load(nf_path, map_location=device)) # load learned weights
    else:
        pass

    # load trained weights for Variational decoder
    decoder_path = folder_name + '/Decoder_model.pt'
    if os.path.exists(decoder_path):
        Decoder.load_state_dict(torch.load(decoder_path, map_location=device)) # load learned weights
    else: 
        pass
    #--------------------------------------------------------------------------------------#

    return Encoder, NF, Decoder
#----------------------------------------------------------------#