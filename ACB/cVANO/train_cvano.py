import tqdm
import functools
from typing import Any, Optional, List,  Dict, Iterable
import numpy as np
import jax
from jax import random, numpy as jnp
import os 
import optax

from typing import Any, Optional
import einops
import flax.linen as nn
import jax.numpy as jnp
from local_attention_flax import LocalAttention

from jax import jit, vmap

from flax.training import train_state
from kymatio.numpy import Scattering2D
from flax import linen as nn 

from jax.flatten_util import ravel_pytree
import matplotlib.pyplot as plt
from jax.nn.initializers import  he_normal
from torch.utils import data
import torch

from jax.tree_util import tree_map
from torch.utils import data


import sys
sys.path.append('../')
import data_loaders as dl

kernel_initialization = he_normal


# def get_free_gpu():
#     os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Used >tmp')
#     memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
#     return str(np.argmin(memory_available))

# os.environ['CUDA_VISIBLE_DEVICES']= get_free_gpu()
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']="False"

plt.rcParams.update(plt.rcParamsDefault)
plt.rc('font', family='serif')
plt.rcParams.update({
                      "text.usetex": True,
                      "font.family": "serif",
                      'text.latex.preamble': r'\usepackage{amsmath}',
                      'font.size': 20,
                      'lines.linewidth': 3,
                      'axes.labelsize': 22,  
                      'axes.titlesize': 24,
                      'xtick.labelsize': 20,
                      'ytick.labelsize': 20,
                      'legend.fontsize': 20,
                      'axes.linewidth': 2})


class DataGenerator(data.Dataset):
    def __init__(self, u, y, s, p,
                 batch_size=100, rng_key=random.PRNGKey(1234)):
        'Initialization'
        self.u = u
        self.y = y
        self.s = s
        self.p = p
        
        self.N = u.shape[0]
        self.batch_size = batch_size
        self.key = rng_key

    def __getitem__(self, index):
        'Generate one batch of data'
        key, subkey = random.split(self.key,2)
        u, y ,s, p = self.__data_generation(subkey)
        return u, y, s, p

    @functools.partial(jit, static_argnums=(0,))
    def __data_generation(self, key):
        'Generates data containing batch_size samples'
        idx = random.choice(key, self.N, (self.batch_size,), replace=False)
        s = self.s[idx,:,:]
        u = self.u[idx,:,:]
        y = self.y[:self.batch_size,:,:]
        p = self.p[idx,:,:]
        return u, y, s, p


# def numpy_collate(batch):
#     if isinstance(batch[0], np.ndarray):
#         return jnp.stack(batch)
#     elif isinstance(batch[0], (tuple,list)):
#         transposed = zip(*batch)
#         return [numpy_collate(samples) for samples in transposed]
#     else:
#         return jnp.array(batch)


def PositionalEncoding(coords, num_octaves=1, start_octave=0):
    num_points, dim = coords.shape
    octaves = jnp.arange(start_octave, start_octave + num_octaves)
    multipliers = 2**octaves * jnp.pi
    while len(multipliers.shape) < len(coords.shape):
        multipliers = jnp.expand_dims(multipliers,axis=0)
    scaled_coords = coords * multipliers
    sines = jnp.sin(scaled_coords).reshape(num_points, dim *  num_octaves)
    cosines = jnp.cos(scaled_coords).reshape(num_points, dim *  num_octaves)
    result = jnp.concatenate((sines, cosines), -1)
    return result

def scatteringTransform(sig, training_batch_size = 100):
    scattering = Scattering2D(J=1, L=3, max_order=2, shape=(32, 32))
    cwtmatr = np.zeros((training_batch_size, 32, 32))
    sig = np.array(sig)
    for i in range(0,training_batch_size):
        scatteringCoeffs = scattering(sig[i,:,:].reshape(32,32))
        cwtmatr[i,:,:] = scatteringCoeffs.reshape(32, 32)
    return cwtmatr

class MLPEncoder(nn.Module):
  encoder_configs: Dict
  @nn.compact
  def __call__(self, u, s):
    u = jnp.concatenate((u,s),axis=-1)
    u = MLP(features=self.encoder_configs["features"])(u)
    sigma = jax.nn.softplus(nn.Dense(features=n)(u))
    # sigma = jnp.exp(nn.Dense(features=n)(u))
    return sigma

class MLPNomad(nn.Module):
  encoder_configs: Dict
  @nn.compact
  def __call__(self, u):
    u = MLP(features=self.encoder_configs["features"])(u)
    u = nn.Dense(features=n)(u)
    return u

class MLPGlobalConditioning(nn.Module):
  encoder_configs: Dict
  @nn.compact
  def __call__(self, u):
    u = MLP(features=self.encoder_configs["features"])(u)
    u = nn.Dense(features=n)(u)
    return u


class MlpBlock(nn.Module):
  mlp_dim: int

  @nn.compact
  def __call__(self, x):
    y = nn.Dense(self.mlp_dim)(x)
    y = nn.gelu(y)
    return nn.Dense(x.shape[-1])(y)


class ConvolutionPrior(nn.Module):
    encoder_configs: Dict
    @nn.compact
    def __call__(self, u):
        u = u.reshape(1, 32,32)
        u = nn.Conv(4,  kernel_size=(2,2), strides=(2,2), padding="SAME")(u)
        u = nn.relu(u)
        u = nn.Conv(8,  kernel_size=(2,2), strides=(2,2), padding="SAME")(u)
        u = nn.relu(u)
        u = nn.Conv(16, kernel_size=(2,2), strides=(2,2), padding="SAME")(u)
        u = nn.relu(u)
        u = nn.Conv(32, kernel_size=(2,2), strides=(2,2), padding="SAME")(u)
        u = nn.relu(u)
        u = u.reshape(1,-1)
        mu = nn.Dense(features=int(n/2))(u)
        sigma = jax.nn.softplus(nn.Dense(features=int(n/2))(u))
        return  mu, sigma

class ConvolutionEncoder(nn.Module):
    encoder_configs: Dict
    @nn.compact
    def __call__(self, u, s):
        s = s.reshape(1, 32,32)
        u = u.reshape(1, 32,32)
        u = jnp.concatenate((u,s),axis=0)
        u = nn.Conv(4,  kernel_size=(2,2), strides=(2,2), padding="SAME")(u)
        u = nn.relu(u)
        u = nn.Conv(8,  kernel_size=(2,2), strides=(2,2), padding="SAME")(u)
        u = nn.relu(u)
        u = nn.Conv(16, kernel_size=(2,2), strides=(2,2), padding="SAME")(u)
        u = nn.relu(u)
        u = nn.Conv(32, kernel_size=(2,2), strides=(2,2), padding="SAME")(u)
        u = nn.relu(u)
        u = u.reshape(1,-1)
        mu = nn.Dense(features=int(n/2))(u)
        sigma = jax.nn.softplus(nn.Dense(features=int(n/2))(u))
        return  mu, sigma


class MLPPrior(nn.Module):
  prior_configs: Dict
  @nn.compact
  def __call__(self, u):
    # u = jnp.concatenate((u,s.reshape(1,P*ds)),axis=-1)
    u = MLP(features=self.prior_configs["features"])(u)
    sigma = jax.nn.softplus(nn.Dense(features=n)(u))
    # sigma = jnp.exp(nn.Dense(features=n)(u))
    mu = nn.Dense(features=n)(u)
    return mu, sigma

class MLP(nn.Module):
  features: List
  @nn.compact
  def __call__(self, x):
    for i, feat in enumerate(self.features):
      x = nn.jit(nn.Dense)(feat)(x)
      if i != len(self.features) - 1:
        x = nn.gelu(x)
    return x

class LayerWiseMLP(nn.Module):
  features: List
  @nn.compact
  def __call__(self, z, y):
    for i, feat in enumerate(self.features):
      y = jnp.concatenate((z[:,i,None],y),axis=-1)
      y = nn.jit(nn.Dense)(feat)(y)
      if i != len(self.features) - 1:
        y = nn.gelu(y)
    return y

class ConcatDecoderMLP(nn.Module):
  decoder_configs: Dict

  @nn.compact
  def __call__(self, z, u, y):
    y = y[None,:]
    u = u.flatten()[None,:]
    if self.decoder_configs["features_conditioning"] is not None:
        u = MLP(features=self.decoder_configs["features_conditioning"])(u)
    else:
        u = jnp.reshape(u,(1,du))
    z = jnp.concatenate((z,y),axis=-1)
    # z = LayerWiseMLP(features=self.decoder_configs["features_decoder"])(z, y)
    z = MLP(features=self.decoder_configs["features_decoder"])(z)
    return z[...,0,:]

class LayerWiseConcatDecoderMLP(nn.Module):
  decoder_configs: Dict
  @nn.compact
  def __call__(self, z, y):
    y = y[None,:]
    y = PositionalEncoding(y, num_octaves=H)
    z = LayerWiseMLP(features=self.decoder_configs["features"])(z, y)
    return z
    
class EncoderSetup(object):
    def __init__(self, encoder_type=None):
        super(EncoderSetup, self).__init__()
        self.encoder_type = encoder_type

    def set_encoder(self, encoder_configs=None):
        return self.encoder_type(encoder_configs)

class PriorSetup(object):
    def __init__(self, prior_type=None):
        super(PriorSetup, self).__init__()
        self.prior_type = prior_type

    def set_prior(self, prior_configs=None):
        return self.prior_type(prior_configs)


class DecoderSetup(object):
    def __init__(self, decoder_type=None):
        super(DecoderSetup, self).__init__()
        self.decoder_type = decoder_type

    def set_decoder(self, decoder_configs=None):
        dec =  nn.vmap(self.decoder_type,
                in_axes=(0,None, None),
                variable_axes={'params': None},
                split_rngs={'params': False}) 
        if decoder_configs["features_conditioning"] is not None:
            dec2 =  nn.vmap(dec,
                    in_axes=(None, None, 0),
                    variable_axes={'params': None},
                    split_rngs={'params': False}) 
        else:
            dec2 =  nn.vmap(dec,
                    in_axes=(None, 0, 0),
                    variable_axes={'params': None},
                    split_rngs={'params': False}) 

        return dec2(decoder_configs)

class NOMAD(nn.Module):
    encoder: nn.Module
    prior: nn.Module
    decoder: nn.Module
    @nn.compact
    def __call__(self, u, y):
        u = jnp.reshape(u, (1, m*du))
        beta = self.encoder(u)
        s_pred = self.decoder(beta[None,:,:], jnp.reshape(u,(m,du)), y)
        return s_pred

class cVANO(nn.Module):
    encoder: nn.Module
    prior: nn.Module
    decoder: nn.Module
    nomad:nn.Module
    @nn.compact
    def __call__(self, u, s, y, p, rng):
        s  = jnp.reshape(s,(1,P*ds))
        u = jnp.reshape(u, (1, m*du))
        mu_e = p
        sigma_e = self.encoder(u, s)
        mu_p, sigma_p = self.prior(u)
        sigma_p = jnp.maximum(sigma_p, 1e-6)
        sigma_e = jnp.maximum(sigma_e, 1e-6)
        beta_e = vmap(self.reparameterization_trick, in_axes=(0,None,None))(rng, mu_e, sigma_e)
        beta_p = vmap(self.reparameterization_trick, in_axes=(0,None,None))(rng, mu_p, sigma_p)
        s_pred_e = self.decoder(beta_e, jnp.reshape(u,(m,du)), y)
        s_pred_p = self.decoder(beta_p, jnp.reshape(u,(m,du)), y)
        return s_pred_e, s_pred_p, s_pred_p, mu_e, sigma_e, mu_p, sigma_p

    def reparameterization_trick(self,rng, mu, sigma):
        """Sample a diagonal Gaussian."""
        return  mu + 1e-2*jnp.sqrt(sigma)*random.normal(rng, mu.shape)
        # return  jnp.exp(mu + jnp.sqrt(sigma)*random.normal(rng, mu.shape))

class generate_samples(nn.Module):
    encoder: nn.Module
    prior: nn.Module
    decoder: nn.Module
    nomad: nn.Module
    @nn.compact
    def __call__(self, u, s, y, p, rng):  
        u = jnp.reshape(u, (1,m*du))
        mu_p, sigma_p = self.prior(u)
        sigma_p = jnp.maximum(sigma_p, 1e-6)
        beta_p = vmap(self.reparameterization_trick, in_axes=(0,None,None))(rng, mu_p, sigma_p)
        s_pred = self.decoder(beta_p, jnp.reshape(u,(m,du)), y)
        return s_pred, mu_p, sigma_p, beta_p

    def reparameterization_trick(self,rng, mu, sigma):
        """Sample a diagonal Gaussian."""
        return  mu + 1e-2*jnp.sqrt(sigma)*random.normal(rng, mu.shape)
        # return  jnp.exp(mu + jnp.sqrt(sigma)*random.normal(rng, mu.shape))


class reconstruct_samples(nn.Module):
    encoder: nn.Module
    prior: nn.Module
    decoder: nn.Module
    @nn.compact
    def __call__(self, u, s, y, p, rng):     
        u = jnp.reshape(u, (1,m*du))
        mu_e = p
        sigma_e = self.encoder(u, s)
        beta_e = vmap(self.reparameterization_trick, in_axes=(0,None,None))(rng, mu_e, sigma_e)
        beta_e = jnp.concatenate((p,beta_e),axis=-1)
        s_pred = self.decoder(beta_e, jnp.reshape(u,(m,du)), y)
        return s_pred, mu_e, sigma_e, beta_e

    def reparameterization_trick(self,rng, mu, sigma):
        """Sample a diagonal Gaussian."""
        return  mu + 1e-2*jnp.sqrt(sigma)*random.normal(rng, mu.shape)
        # return  jnp.exp(mu + jnp.sqrt(sigma)*random.normal(rng, mu.shape))

# class decode_latent_codes(nn.Module):
#     encoder: nn.Module
#     prior: nn.Module
#     decoder: nn.Module
#     @nn.compact
#     def __call__(self,  u, s, y, p, rng):     
#         print(u.shape, p.shape, y.shape)
#         s_pred = self.decoder(p, jnp.reshape(u,(m,du)), y)
#         return s_pred, p


class decode_latent_codes(nn.Module):
    encoder: nn.Module
    prior: nn.Module
    decoder: nn.Module
    @nn.compact
    def __call__(self, u, s, y, p, rng):  
        u = jnp.reshape(u,(m,du))
        print(u.shape, p.shape, y.shape)
        s_pred = self.decoder(p, u, y)
        return s_pred, p

class Loss(object):
    def __init__(self, axis=1, order=None):
        super(Loss, self).__init__()
        self.axis = axis
        self.order = order

    def relative_loss(self, s, s_pred):
        return jnp.power(jnp.linalg.norm(s_pred - s[None,:,:],ord=None,axis=self.axis),2)/jnp.power(jnp.linalg.norm(s[None,:,:],ord=None,axis=self.axis),2)
            
    def gaussian_kl(self, p_mu, p_logsigma, e_mu, e_logsigma):
        """KL divergence from a diagonal Gaussian to the standard Gaussian."""
        # return 0.5*jnp.sum(jnp.log(e_logsigma) - jnp.log(p_logsigma) - 1.0 + (p_mu-e_mu)**2/e_logsigma + p_logsigma/e_logsigma, axis=-1)
        # return 0.5*jnp.sum(e_mu/p_mu + (e_mu - p_mu)**2/p_mu - 1 + jnp.log(e_mu) - jnp.log(e_mu), axis=-1)
        return 0.5*jnp.sum(jnp.log(p_logsigma) - jnp.log(e_logsigma) - 1.0 + ((p_mu-e_mu)**2 + e_logsigma)/p_logsigma, axis=-1)

    def reconstruction_loss(self, s, s_pred):
        return jnp.mean(vmap(self.relative_loss, in_axes=(0,0))(s, s_pred), axis=(1,2))
    
    def lossCVAE(self, s, s_pred, p_mu, p_logsigma, e_mu, e_logsigma):
        return jnp.mean(self.reconstruction_loss(s, s_pred) +\
                        gamma*self.gaussian_kl(p_mu, p_logsigma, e_mu, e_logsigma)[:,0])
    
    def lossGSNN(self, s, s_pred, s_pred_nom):
        return jnp.mean(self.reconstruction_loss(s, s_pred)) #+ self.reconstruction_loss(s, s_pred_nom))

    def __call__(self, s, s_pred_e, s_pred_p, s_pred_nom, p_mu, p_logsigma, e_mu, e_logsigma):
        return  alpha*self.lossCVAE(s, s_pred_e, p_mu, p_logsigma, e_mu, e_logsigma) + (1.-alpha)*self.lossGSNN(s, s_pred_p, s_pred_nom)

lossVAE = Loss()
def calculate_loss_train(state, params, batch, rng):
    u, y, s, p = batch
    elbo_rng = random.split(rng, mc_samples*batch_size)
    elbo_rng = jnp.reshape(elbo_rng,(batch_size, mc_samples,2))
    s_pred_e, s_pred_p, s_pred_nom, p_mu, p_logsigma, e_mu, e_logsigma = vmap(state.apply_fn, in_axes=(None, 0, 0, 0, 0, 0))(params, u, s, y, p, elbo_rng)
    return lossVAE(s, jnp.swapaxes(s_pred_e,1,2), jnp.swapaxes(s_pred_p,1,2), jnp.swapaxes(s_pred_nom,1,2), p_mu, p_logsigma, e_mu, e_logsigma),\
        ( jnp.mean(lossVAE.reconstruction_loss(s,  jnp.swapaxes(s_pred_e,1,2))),
         gamma*jnp.mean(lossVAE.gaussian_kl(p_mu, p_logsigma, e_mu, e_logsigma)[:,0]))

@jax.jit 
def train_step(state, batch, rng):
    grad_fn = jax.value_and_grad(calculate_loss_train, 
                                 argnums=1, 
                                 has_aux=True 
                                )
    (loss_value, aux), grads = grad_fn(state, state.params, batch, rng)
    grads = jax.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), grads)
    lossr, losskl = aux
    state = state.apply_gradients(grads=grads)
    return state, loss_value, lossr, losskl


def train_model(state, data_loader, num_epochs=None, seed=None):
    bar = tqdm.tqdm(np.arange(num_epochs))
    rng = random.PRNGKey(seed)
    rng, key = random.split(rng)
    for epoch in bar:
        count = 0
        for batch in data_loader:
            rng, key = random.split(rng)
            state, loss, lossr, losskl = train_step(state, batch, rng)
            count += 1
            if count==num_train//batch_size:
                break
        bar.set_description("Training Loss: {}, Reconstruction Loss: {}, KL Loss {}".format(loss, lossr, losskl))
    return state

def count_params(params):
    params_flat, _ = ravel_pytree(params)
    print("The number of model parameters is:",params_flat.shape[0])


class load_dataset(object):
    def __init__(self, num_train=1000, num_test=1000, batch_size=100, bounds=None):
        super(load_dataset, self).__init__()
        self.num_train =num_train
        self.num_test  =num_test
        self.batch_size=batch_size

    def dataset_creation(self, number_of_patients = 4374, P=162):
        train_dataset, test_dataset, \
            mins, s_orders, p_min, p_orders,\
                  shape_of_outputs = dl.LoadTurbTowers().get_dataloaders(num_train, number_of_parameters, num_test, batch_size)
        return train_dataset, test_dataset, 1, s_orders, p_orders

number_of_parameters = 6
m = 181
P = 181
dy = 1
du = 20
ds = 20
num_train = 8000
num_test = 1000
epochs = 2000
H = 2
lb = 0
ub = 0.787
mc_samples = 1
lr = 0.001
transition_steps = 1000
decay_rate = 0.99
weight_decay = 0.0001
gamma = 1
#All 5 was best.
decoder_depth = 10
decoder_width = 32
decoder_type  = 0
encoder_depth = 3
encoder_width = 32
encoder_type  = 1
prior_depth   = 3
prior_width   = 32
prior_type    = 1
alpha    = 1.
jitter    = 1e-6
noise    = 0.0
seed    = 6129
n = 6
conditioning_type = 1
batch_size = 100

if encoder_type==0:
    encoder_type =ConvolutionEncoder 
elif encoder_type==1:
    encoder_type =MLPEncoder 
else:
    NotImplementedError("This encoder type is not implemented")

if prior_type==0:
    prior_type =ConvolutionPrior 
elif prior_type==1:
    prior_type =MLPPrior 
else:
    NotImplementedError("This prior type is not implemented")

if decoder_type==0:
    decoder_type =ConcatDecoderMLP 
elif decoder_type==1:
    decoder_type =LayerWiseConcatDecoderMLP 
else:
    NotImplementedError("This decoder type is not implemented")


ld = load_dataset(num_train=num_train, num_test=num_test, batch_size=batch_size, bounds = (ub, lb))
train_data_loader, test_data, U_ordes, s_orders, p_orders = ld.dataset_creation(P=P)

print("Data Loaded")
key1, key2, key3, key4 = random.split(random.PRNGKey(seed), 4)

u = random.uniform(key1, (1, m*du))
s = random.uniform(key1, (P,ds))
y = random.uniform(key2, (P,dy))
p = random.uniform(key2, (1, n))
rng = random.split(key3, mc_samples)

mlp_decoder_architecture = []
for i in range(decoder_depth):
    if i == decoder_depth-1:
        mlp_decoder_architecture.append(ds)
    else:
        mlp_decoder_architecture.append(decoder_width)

mlp_encoder_architecture = []
for i in range(encoder_depth):
    if i == encoder_depth-1:
        mlp_encoder_architecture.append(n)
    else:
        mlp_encoder_architecture.append(encoder_width)

mlp_prior_architecture = []
for i in range(prior_depth):
    if i == prior_depth-1:
        mlp_prior_architecture.append(n)
    else:
        mlp_prior_architecture.append(prior_width)

if conditioning_type==0:
    mlp_coditioning_architecture = []
    for i in range(encoder_depth):
        if i == encoder_depth-1:
            mlp_coditioning_architecture.append(du)
        else:
            mlp_coditioning_architecture.append(encoder_width)
elif conditioning_type==1:
    mlp_coditioning_architecture = None 
else:
    NotImplementedError("This decoder type is not implemented")

print("Model Initialized")
encoder = EncoderSetup(encoder_type=encoder_type).set_encoder(encoder_configs={"features":mlp_encoder_architecture})
encoder_nomad = EncoderSetup(encoder_type=MLPNomad).set_encoder(encoder_configs={"features":[128, 128, 128, n]})
prior  = PriorSetup(prior_type=prior_type).set_prior(prior_configs={"features":mlp_prior_architecture})
decoder = DecoderSetup(decoder_type=decoder_type).set_decoder(decoder_configs={"features_decoder":mlp_decoder_architecture, "features_conditioning":mlp_coditioning_architecture})
nomad = NOMAD(encoder=encoder_nomad, prior=prior, decoder=decoder)
model = cVANO(encoder=encoder, prior=prior, decoder=decoder, nomad=nomad)
params = model.init({'params': key4}, u, s, y, p, rng)

count_params(params)

del u, y, s

exponential_decay_scheduler = optax.exponential_decay(init_value=lr, transition_steps=transition_steps,
                                                      decay_rate=decay_rate, transition_begin=0,
                                                      staircase=False)

optimizer = optax.adamw(learning_rate=exponential_decay_scheduler, weight_decay=weight_decay)

model_state = train_state.TrainState.create(apply_fn=model.apply,
                                            params=params,
                                            tx=optimizer)

trained_model_state = train_model(model_state, train_data_loader, num_epochs=epochs, seed=seed)


N_samples = 100
multiplier = 1
batch = batch_size
model_sample = generate_samples(encoder=encoder, prior=prior, decoder=decoder, nomad=nomad)

elbo_rng = random.split(random.PRNGKey(seed), N_samples*batch*multiplier)
s_pred_all = np.zeros((num_test, P, N_samples*multiplier, ds))
z_e_all    = np.zeros((num_test, N_samples*multiplier, 1, number_of_parameters))
s_test_all = np.zeros((num_test, P, ds))
for b in range(num_test//batch_size):
    U_test, y_test, s_test, p_test = next(iter(test_data)) 
    for i in range(0, multiplier):
        elbo_rng_i = jnp.reshape(elbo_rng[i*N_samples*batch:(i+1)*N_samples*batch, :],(batch, N_samples, 2))
        s_pred, mu_e, sigma_e, beta_e = vmap(model_sample.apply, in_axes=(None, 0, 0, 0, 0, 0))(trained_model_state.params, U_test, s_test, y_test,  p_test, elbo_rng_i)
        z_e = vmap(vmap(model.reparameterization_trick, in_axes=(0,None,None)),in_axes=(0,0,0))(elbo_rng_i, mu_e, sigma_e)
        s_pred_all[b*batch_size:(b+1)*batch_size, :, i*N_samples:(i+1)*N_samples, :] = s_pred
        s_test_all[b*batch_size:(b+1)*batch_size, :, :] = s_test
        z_e_all[b*batch_size:(b+1)*batch_size, i*N_samples:(i+1)*N_samples, :, :] = z_e
        del z_e, s_pred

z_e = z_e_all
s_pred = s_pred_all

del z_e_all, s_pred_all

mean_s_pred = jnp.mean(s_pred, axis=2)

rl2_error = jnp.linalg.norm(s_test_all - mean_s_pred, ord=None, axis=1)/jnp.linalg.norm(s_test_all, ord=None, axis=1)

mean_acc = jnp.mean(rl2_error,axis=0)
std_acc  = jnp.std( rl2_error,axis=0)
max_acc  = jnp.max( rl2_error,axis=0)
min_acc  = jnp.min( rl2_error,axis=0)

def crps_per_parameter(y_pred, y_true, sample_weight=None):
    
    num_samples = y_pred.shape[0]
    absolute_error = jnp.mean(jnp.abs(y_pred - y_true), axis=0)

    if num_samples == 1:
        return jnp.average(absolute_error, weights=sample_weight)

    y_pred = jnp.sort(y_pred, axis=0)
    b0 = y_pred.mean(axis=0)
    b1_values = y_pred * jnp.arange(num_samples).reshape((num_samples, 1))
    b1 = b1_values.mean(axis=0) / num_samples

    per_obs_crps = absolute_error + b0 - 2 * b1
    return per_obs_crps

crps_pp = np.zeros((p_test.shape[0], p_test.shape[1]))
for i in range(p_test.shape[0]):
    crps_pp = crps_per_parameter(z_e[i,:,0,:], p_test[i,:])

print(crps_pp)

for i in range(0,mean_acc.shape[0]):
    print(f"Mean function test error for pressure at vessel {i:d}: {mean_acc[i]:1.8f}, Std function test error: {std_acc[i]:1.8f}, Min function test error: {min_acc[i]:1.8f}, Max function test error: {max_acc[i]:1.8f}" )

mean_error_across = jnp.linalg.norm(s_test_all - mean_s_pred, ord=None)/jnp.linalg.norm(s_test_all, ord=None)
print(f"Mean function test error across vessels: {mean_error_across:1.8f} " )

np.savez_compressed('turb_towers_results_using_infered_parameters.npz', true_pressures=s_test_all, pressures=s_pred, parameters=z_e)
print("saved results")

model_sample = decode_latent_codes(encoder=encoder, prior=prior, decoder=decoder)

N_samples  = 1
multiplier = 1
s_pred_all = np.zeros((num_test, P, N_samples*multiplier, ds))
z_e_all    = np.zeros((num_test, N_samples*multiplier, 1, number_of_parameters))
s_test_all = np.zeros((num_test, P, ds))
for b in range(num_test//batch_size):
    U_test, y_test, s_test, p_test = next(iter(test_data)) 
    p = p_test[:, None, None,...]
    for i in range(0, multiplier):
        elbo_rng_i = jnp.reshape(elbo_rng[i*N_samples*batch:(i+1)*N_samples*batch, :],(batch, N_samples, 2))
        s_pred, beta_e = vmap(model_sample.apply, in_axes=(None, 0, 0, 0, 0, 0))(trained_model_state.params, U_test, s_test, y_test,  p, elbo_rng_i)
        s_pred_all[b*batch_size:(b+1)*batch_size, :, i*N_samples:(i+1)*N_samples, :] = s_pred
        s_test_all[b*batch_size:(b+1)*batch_size, :, :] = s_test
        z_e_all[b*batch_size:(b+1)*batch_size, i*N_samples:(i+1)*N_samples, :, :] = beta_e
        del s_pred

mean_s_pred = jnp.mean(s_pred_all, axis=2)

rl2_error = jnp.linalg.norm(s_test_all - mean_s_pred, ord=None, axis=1)/jnp.linalg.norm(s_test_all, ord=None, axis=1)

mean_acc = jnp.mean(rl2_error,axis=0)
std_acc  = jnp.std( rl2_error,axis=0)
max_acc  = jnp.max( rl2_error,axis=0)
min_acc  = jnp.min( rl2_error,axis=0)

for i in range(0,mean_acc.shape[0]):
    print(f"Mean function test error for pressure at vessel {i:d}: {mean_acc[i]:1.8f}, Std function test error: {std_acc[i]:1.8f}, Min function test error: {min_acc[i]:1.8f}, Max function test error: {max_acc[i]:1.8f}" )

ze_mean = jnp.mean(z_e,axis=1)

mean_error_across = jnp.linalg.norm(s_test_all - mean_s_pred, ord=None)/jnp.linalg.norm(s_test_all, ord=None)
print(f"Mean function test error across vessels: {mean_error_across:1.8f}" )

np.savez_compressed('turb_towers_results_decoding_latent_vectors_of_true_parameters.npz',true_pressures=s_test_all, pressures=s_pred_all, parameters=z_e_all)
print("saved results")

# ----------------------------------------------------------- #
# ----------------------------------------------------------- #
# ----------------------------------------------------------- #
# ----------------------------------------------------------- #

from torch.utils import data
from jax.tree_util import tree_map

inputs = torch.load('ood_turbtowers_dataset.pt')

def numpy_collate(batch):
  return tree_map(np.asarray, data.default_collate(batch))

class NumpyLoader(data.DataLoader):
  def __init__(self, dataset, batch_size=1,
                shuffle=False, sampler=None,
                batch_sampler=None, num_workers=0,
                pin_memory=False, drop_last=False,
                timeout=0, worker_init_fn=None):
    super(self.__class__, self).__init__(dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=numpy_collate,
        pin_memory=pin_memory,
        drop_last=drop_last,
        timeout=timeout,
        worker_init_fn=worker_init_fn)

test_data_ood = NumpyLoader(inputs,batch_size=100,shuffle=False)

N_samples = 100
multiplier = 1
batch = batch_size
model_sample = generate_samples(encoder=encoder, prior=prior, decoder=decoder, nomad=nomad)

elbo_rng = random.split(random.PRNGKey(seed), N_samples*batch*multiplier)
s_pred_all = np.zeros((100, P, N_samples*multiplier, ds))
z_e_all    = np.zeros((100, N_samples*multiplier, 1, number_of_parameters))
s_test_all = np.zeros((100, P, ds))
for b in range(1):
    data_from_loader = next(iter(test_data_ood)) 
    U_test,  s_test, p_test = jnp.swapaxes(data_from_loader[1],1,2), jnp.swapaxes(data_from_loader[1],1,2), data_from_loader[0]
    for i in range(0, multiplier):
        elbo_rng_i = jnp.reshape(elbo_rng[i*N_samples*batch:(i+1)*N_samples*batch, :],(batch, N_samples, 2))
        s_pred, mu_e, sigma_e, beta_e = vmap(model_sample.apply, in_axes=(None, 0, 0, 0, 0, 0))(trained_model_state.params, U_test, s_test, y_test,  p_test, elbo_rng_i)
        z_e = vmap(vmap(model.reparameterization_trick, in_axes=(0,None,None)),in_axes=(0,0,0))(elbo_rng_i, mu_e, sigma_e)
        s_pred_all[b*batch_size:(b+1)*batch_size, :, i*N_samples:(i+1)*N_samples, :] = s_pred
        s_test_all[b*batch_size:(b+1)*batch_size, :, :] = s_test
        z_e_all[b*batch_size:(b+1)*batch_size, i*N_samples:(i+1)*N_samples, :, :] = z_e
        del z_e, s_pred

z_e = z_e_all
s_pred = s_pred_all

del z_e_all, s_pred_all

mean_s_pred = jnp.mean(s_pred, axis=2)

rl2_error = jnp.linalg.norm(s_test_all - mean_s_pred, ord=None, axis=1)/jnp.linalg.norm(s_test_all, ord=None, axis=1)

mean_acc = jnp.mean(rl2_error,axis=0)
std_acc  = jnp.std( rl2_error,axis=0)
max_acc  = jnp.max( rl2_error,axis=0)
min_acc  = jnp.min( rl2_error,axis=0)

def crps_per_parameter(y_pred, y_true, sample_weight=None):
    
    num_samples = y_pred.shape[0]
    absolute_error = jnp.mean(jnp.abs(y_pred - y_true), axis=0)

    if num_samples == 1:
        return jnp.average(absolute_error, weights=sample_weight)

    y_pred = jnp.sort(y_pred, axis=0)
    b0 = y_pred.mean(axis=0)
    b1_values = y_pred * jnp.arange(num_samples).reshape((num_samples, 1))
    b1 = b1_values.mean(axis=0) / num_samples

    per_obs_crps = absolute_error + b0 - 2 * b1
    return per_obs_crps

crps_pp = np.zeros((p_test.shape[0], p_test.shape[1]))
for i in range(p_test.shape[0]):
    crps_pp = crps_per_parameter(z_e[i,:,0,:], p_test[i,:])

print(crps_pp)

for i in range(0,mean_acc.shape[0]):
    print(f"Mean function test error for pressure at vessel {i:d}: {mean_acc[i]:1.8f}, Std function test error: {std_acc[i]:1.8f}, Min function test error: {min_acc[i]:1.8f}, Max function test error: {max_acc[i]:1.8f}" )

mean_error_across = jnp.linalg.norm(s_test_all - mean_s_pred, ord=None)/jnp.linalg.norm(s_test_all, ord=None)
print(f"Mean function test error across vessels: {mean_error_across:1.8f} " )

np.savez_compressed('turb_towers_results_using_infered_parameters_outofdistribution.npz', true_pressures=s_test_all, pressures=s_pred, parameters=z_e)
print("saved results")

model_sample = decode_latent_codes(encoder=encoder, prior=prior, decoder=decoder)

N_samples  = 1
multiplier = 1
s_pred_all = np.zeros((100, P, N_samples*multiplier, ds))
z_e_all    = np.zeros((100, N_samples*multiplier, 1, number_of_parameters))
s_test_all = np.zeros((100, P, ds))
for b in range(1):
    data_from_loader = next(iter(test_data_ood)) 
    U_test,  s_test, p_test = jnp.swapaxes(data_from_loader[1],1,2), jnp.swapaxes(data_from_loader[1],1,2), data_from_loader[0]
    p = p_test[:, None, None,...]
    for i in range(0, multiplier):
        elbo_rng_i = jnp.reshape(elbo_rng[i*N_samples*batch:(i+1)*N_samples*batch, :],(batch, N_samples, 2))
        s_pred, beta_e = vmap(model_sample.apply, in_axes=(None, 0, 0, 0, 0, 0))(trained_model_state.params, U_test, s_test, y_test,  p, elbo_rng_i)
        s_pred_all[b*batch_size:(b+1)*batch_size, :, i*N_samples:(i+1)*N_samples, :] = s_pred
        s_test_all[b*batch_size:(b+1)*batch_size, :, :] = s_test
        z_e_all[b*batch_size:(b+1)*batch_size, i*N_samples:(i+1)*N_samples, :, :] = beta_e
        del s_pred

mean_s_pred = jnp.mean(s_pred_all, axis=2)

rl2_error = jnp.linalg.norm(s_test_all - mean_s_pred, ord=None, axis=1)/jnp.linalg.norm(s_test_all, ord=None, axis=1)

mean_acc = jnp.mean(rl2_error,axis=0)
std_acc  = jnp.std( rl2_error,axis=0)
max_acc  = jnp.max( rl2_error,axis=0)
min_acc  = jnp.min( rl2_error,axis=0)

for i in range(0,mean_acc.shape[0]):
    print(f"Mean function test error for pressure at vessel {i:d}: {mean_acc[i]:1.8f}, Std function test error: {std_acc[i]:1.8f}, Min function test error: {min_acc[i]:1.8f}, Max function test error: {max_acc[i]:1.8f}" )

ze_mean = jnp.mean(z_e,axis=1)

mean_error_across = jnp.linalg.norm(s_test_all - mean_s_pred, ord=None)/jnp.linalg.norm(s_test_all, ord=None)
print(f"Mean function test error across vessels: {mean_error_across:1.8f}" )

np.savez_compressed('turb_towers_results_decoding_latent_vectors_of_true_parameters_outofdistribution.npz',true_pressures=s_test_all, pressures=s_pred_all, parameters=z_e_all)
print("saved results")
