import numpy as np
import matplotlib.pyplot as plt

from keras.layers import *
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.models import Model
from tools.utils import load_cmu_small

rng = np.random.RandomState(23455)

latent_dim = 2
intermediate_dim = 10
batch_size = 1
epsilon_std = 0.01
nb_epoch = 1

datasets, std, mean = load_cmu_small(rng)

# Shape = (MB, 240, 66)
x_train = datasets[0][0].swapaxes(1, 2)

input_motion = Input(batch_shape=(batch_size, x_train.shape[1], x_train.shape[2]))

# shape = (240, 64)
x = Convolution1D(64, 25, activation='relu', border_mode='same')(input_motion)
# shape = (120, 64)
x = MaxPooling1D(pool_length=2, stride=None)(x)
#x = Convolution1D(128, 25, border_mode='same')(x)
#x = MaxPooling1D(pool_length=2, stride=None)(x)
#x = Convolution1D(256, 25, border_mode='same')(x)
#x = MaxPooling1D(pool_length=2, stride=None)(x)

# shape = (120 * 64)
x = Flatten()(x)

# shape = (128)
h = Dense(intermediate_dim, activation='relu')(x)

## Estimate mean and std dev of a Gaussian

# shape = (10)
z_mean = Dense(latent_dim)(h)
z_log_std = Dense(latent_dim)(h)

def sampling(args):
    z_mean, z_log_std = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim),
                              mean=0., std=epsilon_std)
    return z_mean + K.exp(z_log_std) * epsilon

# shape = (10)
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_std])
# shape = (128)
x = Dense(intermediate_dim, activation='relu')(z)

# shape = (120*64)
x = Dense(120*64)(x)
# shape = (120, 64)
x = Reshape((120, 64))(x)

#x = UpSampling1D(length=2)(x)
#x = Convolution1D(256, 25, border_mode='same')(x)
#x = UpSampling1D(length=2)(x)
#x = Convolution1D(128, 25, border_mode='same')(x)
#x = UpSampling1D(length=2)(x)
#x = Convolution1D(64, 25, activation='relu', border_mode='same')(x)

# shape = (240, 64)
x = UpSampling1D(length=2)(x)
# shape = (240, 66)
x = Convolution1D(66, 25, activation='linear', border_mode='same')(x)

def vae_loss(input_motion, x):
    xent_loss = objectives.mse(input_motion, x)
    kl_loss = - 0.5 * K.mean(1 + z_log_std - K.square(z_mean) - K.exp(z_log_std), axis=-1)

    kl_loss_func = K.function([z_log_std], [kl_loss])
    #print kl_loss
    
    #return xent_loss + kl_loss
    return xent_loss

vae = Model(input_motion, x)
vae.compile(optimizer='rmsprop', loss=vae_loss)

get_layer_output = K.function([vae.layers[0].input], [vae.layers[0].output])
layer_output = get_layer_output([x_train])[0]
print layer_output.shape

get_layer_output = K.function([vae.layers[0].input], [vae.layers[1].output])
layer_output = get_layer_output([x_train])[0]
print layer_output.shape

get_layer_output = K.function([vae.layers[0].input], [vae.layers[2].output])
layer_output = get_layer_output([x_train])[0]
print layer_output.shape

vae.fit(x_train, x_train,
        shuffle=True,
        nb_epoch=nb_epoch,
        batch_size=batch_size)