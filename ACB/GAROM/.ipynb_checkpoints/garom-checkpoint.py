import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pdb
from timeit import default_timer
import sys
sys.path.append('../')
from utilities import *


# simple Generator Network
class Generator(nn.Module):
    def __init__(self, input_dimension, parameters_dimension,
                 noise_dimension, activation=torch.nn.SiLU):
        super().__init__()

        self._noise_dimension = noise_dimension
        self._activation = activation

        self.lifting = torch.nn.Linear(1, 20)
        
        self.model = torch.nn.Sequential(
            torch.nn.Linear(6 * self._noise_dimension, input_dimension // 6),
            self._activation(),
            torch.nn.Linear(input_dimension // 6, input_dimension // 3),
            self._activation(),
            torch.nn.Linear(input_dimension // 3, input_dimension)
        )
        

        
        self.condition = torch.nn.Sequential(
            torch.nn.Linear(parameters_dimension, 2 * self._noise_dimension),
            self._activation(),
            torch.nn.Linear(2 * self._noise_dimension, 5 * self._noise_dimension)
        )

    def forward(self, param):
        # uniform sampling in [-1, 1]
        z = torch.rand(size=(param.shape[0], self._noise_dimension),
                       device=param.device,
                       dtype=param.dtype,
                       requires_grad=True)
        z = 2. * z - 1.

        # conditioning by concatenation of mapped parameters
        input_ = torch.cat((z, self.condition(param)), dim=-1)
        lifted_input_ = self.lifting(input_[:,:,None])
        lifted_input_ = lifted_input_.permute(0,2,1)
        out = self.model(lifted_input_)

        return out


# Simple Discriminator Network
class Discriminator(nn.Module):
    def __init__(self, input_dimension, parameter_dimension,
                 hidden_dimension, activation=torch.nn.ReLU):
        super().__init__()

        self._activation = activation


        self.lifting = torch.nn.Linear(1, 20)
        
        self.encoding = torch.nn.Sequential(
            torch.nn.Linear(input_dimension, input_dimension // 3),
            self._activation(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(input_dimension // 3, input_dimension // 6),
            self._activation(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(input_dimension // 6, hidden_dimension)
        )
        self.decoding = torch.nn.Sequential(
            torch.nn.Linear(2*hidden_dimension, input_dimension // 6),
            self._activation(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(input_dimension // 6, input_dimension // 3),
            self._activation(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(input_dimension // 3, input_dimension),
        )

        self.condition = torch.nn.Sequential(
            torch.nn.Linear(parameter_dimension, hidden_dimension // 2),
            self._activation(),
            torch.nn.Linear(hidden_dimension // 2, hidden_dimension)
        )
       
    def forward(self, y, theta):
        encoding = self.encoding(y)
        lifted_theta = self.lifting(theta[:,:,None])
        lifted_theta = lifted_theta.permute(0,2,1)
        conditioning = torch.cat((encoding, self.condition(lifted_theta)), dim=-1)
        decoding = self.decoding(conditioning)
        return decoding


class GAROM(object):

    def __init__(self, input_dimension, hidden_dimension,
                 parameters_dimension, noise_dimension,
                 regularizer=0.,
                 optimizers=None,
                 optimizer_kwargs=None,
                 schedulers=None,
                 scheduler_kwargs=None):
        """GAROM base class implementation.

        :param input_dimension: input vector dimension
        :type input_dimension: int
        :param hidden_dimension: latent dimension for the discriminator
        :type hidden_dimension: int
        :param parameters_dimension: input parameter dimension
        :type parameters_dimension: int
        :param noise_dimension: dimension of the noise vector
        :type noise_dimension: int
        :param regularizer: regularizer term possible values {0, 1}, defaults to 0.
            The behaviour for other values is not tested.
        :type regularizer: float, optional
        :param optimizers: a dictionary with keys  'generator'
            and 'discriminator', and values the generator and
            discriminator torch optimizer respectively,
            defaults to None
        :type optimizers: dict, optional
        :param optimizer_kwargs: a dictionary with keys  'generator'
            and 'discriminator', and values the dictionary containing
            the keywords arguments for the optimizers, defaults to None
        :type optimizer_kwargs: dict, optional
        :param schedulers:  a dictionary with keys  'generator'
            and 'discriminator', and values the generator and
            discriminator torch scheduler respectively,
            defaults to None
        :type schedulers: dict, optional
        :param scheduler_kwargs: a dictionary with keys  'generator'
            and 'discriminator', and values the dictionary containing
            the keywords arguments for the schedulers, defaults to None
        :type scheduler_kwargs: dict, optional


        :Example:
            >>> optimizers = {'generator': torch.optim.Adam,
                              'discriminator': torch.optim.Adam}
            >>> optimizers_kwds = {'generator': {"lr": 0.001},
                                   'discriminator': {"lr": 0.001}}
            >>> schedulers = {'generator': torch.optim.lr_scheduler.StepLR,
                              'discriminator': torch.optim.lr_scheduler.StepLR}
            >>> schedulers_kwds = {'generator': {"gamma": 0.8,
                                                 "step_size":step_size},
                                    'discriminator': {"gamma": 0.8,
                                                      "step_size":step_size}}
            >>> garom = GAROM(input_dimension=1024,
                              hidden_dimension=64,
                              parameters_dimension=2,
                              noise_dimension=12,
                              regularizer=1,
                              optimizers=optimizers,
                              optimizer_kwargs=optimizers_kwds,
                              schedulers=schedulers,
                              scheduler_kwargs=schedulers_kwds)
        """

        # setting generator
        self._generator = Generator(input_dimension=input_dimension,
                                    parameters_dimension=parameters_dimension,
                                    noise_dimension=noise_dimension)

        # setting discriminator
        self._discriminator = Discriminator(
            input_dimension=input_dimension,
            parameter_dimension=parameters_dimension,
            hidden_dimension=hidden_dimension)

        # keys for optimizer and scheduler
        self._keys = ["generator", "discriminator"]

        # set the default optimizer
        if optimizers is None:
            optimizers = {"generator": torch.optim.Adam,
                          "discriminator": torch.optim.Adam}

        # set default  optimizer hyperparameters
        if optimizer_kwargs is None:
            optimizer_kwargs = {"generator": {"lr": 0.001},
                                "discriminator": {"lr": 0.001}}

        # setting optimizers
        self._gen_optim = optimizers["generator"](
            self._generator.parameters(),
            **optimizer_kwargs["generator"])
        self._disc_optim = optimizers["discriminator"](
            self._discriminator.parameters(),
            **optimizer_kwargs["discriminator"])

        # set the scheduler
        self._gen_sched = None
        self._disc_sched = None
        if schedulers is not None:
            self._gen_sched = schedulers["generator"](
                self._gen_optim,
                **scheduler_kwargs["generator"])
            self._disc_sched = schedulers["discriminator"](
                self._disc_optim,
                **scheduler_kwargs["discriminator"])

        # set denoising
        if isinstance(regularizer, float):
            self._regularizer = regularizer
        else:
            raise ValueError('Expected a float of value {0, 1}')

    def save(self, generator_name, discriminator_name):
        """Saving GAROM model

        :param generator_name: file name to save generator
        :type generator_name: str
        :param discriminator_name: file name to save generator
        :type discriminator_name: str
        """
        torch.save(self._generator.state_dict(), generator_name)
        torch.save(self._discriminator.state_dict(), discriminator_name)

    def load(self, generator_name, discriminator_name, **kwargs):
        """Loading GAROM model

        :param generator_name: path where the generator is saved
        :type generator_name: str
        :param discriminator_name: path where the generator is saved
        :type discriminator_name: str
        """
        self._generator.load_state_dict(
            torch.load(generator_name, **kwargs))
        self._discriminator.load_state_dict(
            torch.load(discriminator_name, **kwargs))

    def to(self, *args, **kwargs):
        """Sending GAROM to device, see Torch.to"""
        self._generator.to(*args, **kwargs)
        self._discriminator.to(*args, **kwargs)

    def eval(self):
        """GAROM evaluation mode"""
        self._generator.eval()
        self._discriminator.eval()

    def sample(self, condition):
        """Sample routine for GAROM

        :param condition: condition to obtain the sample from
        :type condition: torch.tensor
        :return: solution sample
        :rtype: torch.tensor
        """
        return self._generator(condition)

    def estimate(self, condition, mc_steps=20, variance=False):
        """Estimating solution via predicting distribution.

        :param condition: condition to obtain the sample from
        :type condition: torch.tensor
        :param mc_steps: number of montecarlo steps, defaults to 20
        :type mc_steps: int, optional
        :param variance: variable to return also the variance,
            defaults to False
        :type variance: bool, optional
        :return: predictive distribution, and variance if `variance=True`
        :rtype: torch.tensor
        """
        field_sample = [self.sample(condition) for _ in range(mc_steps)]
        field_sample = torch.stack(field_sample)

        # extract mean
        mean = field_sample.mean(dim=0)

        if variance:
            var = field_sample.var(dim=0)
            return mean, var

        return mean

    @property
    def generator(self):
        return self._generator

    @property
    def discriminator(self):
        return self._discriminator

    def train(self, train_loader, test_loader, epochs=1000, gamma=0.75,
              lambda_k=0.001, every=None, save_csv=None):
        """Training procedure for GAROM.

        :param dataloader: dataloader for training
        :type dataloader: torch.DataLoader
        :param epochs: number of training epochs, defaults to 1000
        :type epochs: int, optional
        :param gamma: gamma value for BEGAN, defaults to 0.75
        :type gamma: float, optional
        :param lambda_k: lambda_k for BEGAN, defaults to 0.001
        :type lambda_k: float, optional
        :param every: variable to print on screen, defaults to None
        :type every: int, optional
        :param save_csv: path for saving as csv, defaults to None
        :type save_csv: str, optional
        """
        # print every?
        if every is None:
            every = epochs + 1

        # train began
        self._train_began(train_loader=train_loader,
                          test_loader=test_loader,
                          generator=self._generator,
                          discriminator=self._discriminator,
                          optimizer_G=self._gen_optim,
                          optimizer_D=self._disc_optim,
                          gamma=gamma,
                          lambda_k=lambda_k,
                          epochs=epochs,
                          every=every,
                          regularizer=self._regularizer,
                          save_csv=save_csv
                          )

    def _train_began(self, train_loader, test_loader, generator, discriminator,
                     optimizer_G, optimizer_D, scheduler_G=None,
                     scheduler_D=None, gamma=0.75, lambda_k=0.001,
                     epochs=10000, every=1000, regularizer=0., save_csv=None):

        # BEGAN hyper parameter for control theory
        k = 0.0

        # chosing device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # sending to device networks
        generator.to(device)
        discriminator.to(device)

        # for saving in csv
        columns=['epoch', 'g_loss', 'd_loss', 'residual', 'metric']

        # choose the loss
        loss_began = F.l1_loss
        
        
        train_loss_l1 = torch.zeros(int(np.ceil(8000/64)))
        test_loss_l1 = torch.zeros(1000)
        test_loss_l2 = torch.zeros(1000)
        
        best_model = 10
        
        # training starts
        for epoch in range(epochs):
            t1 = default_timer()
            generator.train()
            for iter, data in enumerate(train_loader):
                # Adversarial ground truths
                theta, y, yy = data
                theta = theta.to(device)
                y = y.to(device)

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Generate a batch of images
                gen_imgs = generator(theta)

                # Discriminator pass
                d_real = discriminator(y, theta)
                d_fake = discriminator(gen_imgs.detach(), theta) 

                d_loss_real = loss_began(d_real, y)
                d_loss_fake = loss_began(d_fake, gen_imgs.detach())

                d_loss = d_loss_real - k * d_loss_fake 
                
                optimizer_D.zero_grad()
                d_loss.backward()
                optimizer_D.step()

                # -----------------
                #  Train Generator
                # -----------------

                # Generate a batch of images
                gen_imgs = generator(theta)

                r_loss = F.l1_loss(y, gen_imgs)
                d_fake = discriminator(gen_imgs, theta)
                g_loss = loss_began(d_fake,gen_imgs) + regularizer * r_loss + 100 * h1_loss(gen_imgs, y)
                
                train_loss_l1[iter] = torch.mean(relative_error(gen_imgs, y, p=1)).item()

                optimizer_G.zero_grad()
                g_loss.backward()
                optimizer_G.step()

                # ----------------
                # Update weights
                # ----------------
                diff = torch.mean(gamma * d_loss_real - d_loss_fake)

                # Update weight term for fake samples
                k = k + lambda_k * diff.item()

                k = min(max(k, 0), 1)  # Constraint to interval [0, 1]

                # Update convergence metric
                M = (d_loss_real + torch.abs(diff))
                
                
            
            if epoch % every == every - 1:
                print(f"Epoch {epoch}: training time: {default_timer() - t1}")
            print(f'L1 train error: {torch.mean(train_loss_l1).item():.4f}% +/- {torch.std(train_loss_l1).item():.4f}% ')
                
            with torch.no_grad():
                generator.eval()
                for iter, data in enumerate(test_loader):
                    # Adversarial ground truths
                    theta, y, _ = data
                    theta = theta.to(device)
                    y = y.to(device)

                    # Generate a batch of images
                    gen_imgs = generator(theta)

                    test_loss_l1[iter] = torch.mean(relative_error(gen_imgs, y, p=1))
                    test_loss_l2[iter] = torch.mean(relative_error(gen_imgs, y, p=2))

                
            if epoch % every == every - 1:
                print(f'L1 test error: {torch.mean(test_loss_l1).item():.4f}% +/- {torch.std(test_loss_l1).item():.4f}% ')
                print(f'L2 test error: {torch.mean(test_loss_l2).item():.4f}% +/- {torch.std(test_loss_l2).item():.4f}% ')
                print('\n')
                
            test_error = torch.mean(test_loss_l1).item()
            if test_error < best_model:
                best_model = test_error
                print(f"Saving with {test_error:.4f} error")
                torch.save(generator, "GAROM_TurbulenceTowers.pt")

            if scheduler_G is not None:
                scheduler_G.step()

            if scheduler_D is not None:
                scheduler_D.step()
 
