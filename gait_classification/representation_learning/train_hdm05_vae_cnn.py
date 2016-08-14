'''This script demonstrates how to build a variational autoencoder with Keras.
Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
'''
import numpy as np
import matplotlib.pyplot as plt 
import sys

from nn.AnimationPlotLines import animation_plot

#np.random.seed(int(sys.argv[1]))

from keras.layers import *
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.callbacks import EarlyStopping
from keras.datasets import mnist
from keras.optimizers import Nadam
from tools.utils import load_hdm05_easy, load_cmu, load_locomotion, load_cmu_small

rng = np.random.RandomState(23455)

datasets, std, mean = load_locomotion(rng)
#datasets_ft = load_hdm05_easy(rng)

#x_train = np.concatenate([datasets_ft[0][0][:1900], datasets_ft[1][0][:610],
#                          datasets_ft[2][0][:610], datasets[0][0][:17920]])

x_train = datasets[0][0][:320]
x_train = x_train.swapaxes(1, 2)

print x_train.shape

I = np.arange(len(x_train))
rng.shuffle(I)
x_train = x_train[I]

batch_size = 10
original_dim = 66*240
latent_dim = 100
intermediate_dim = 500
epsilon_std = 0.01
nb_epoch = 150

input_motion = Input(batch_shape=(batch_size, x_train.shape[1], x_train.shape[2]))

# shape = (240, 64)
x = Convolution1D(64, 25, border_mode='same')(input_motion)
x = Activation('relu')(x)
# shape = (120, 64)
x = MaxPooling1D(pool_length=2, stride=None)(x)
# shape = (120, 128)
x = Convolution1D(128, 25, border_mode='same')(x)
x = Activation('relu')(x)
# shape = (60, 128)
x = MaxPooling1D(pool_length=2, stride=None)(x)
# shape = (60, 256)
x = Convolution1D(256, 25, border_mode='same')(x)
x = Activation('relu')(x)
# shape = (30, 256)
x = MaxPooling1D(pool_length=2, stride=None)(x)

# shape = (30*256)
x = Flatten()(x)

# shape = (128)
h = Dense(intermediate_dim, activation='relu')(x)

z_mean = Dense(latent_dim)(h)
z_log_std = Dense(latent_dim)(h)

def sampling(args):
    z_mean, z_log_std = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim),
                              mean=0., std=epsilon_std)
    return z_mean + K.exp(z_log_std) * epsilon

# shape = (50)
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_std])
# shape = (128)
x = Dense(intermediate_dim, activation='relu')(z)

x = Dense(30*256)(x)
x = Reshape((30, 256))(x)

x = UpSampling1D(length=2)(x)
x = Convolution1D(256, 25, activation='linear', border_mode='same')(x)
x = UpSampling1D(length=2)(x)
x = Convolution1D(128, 25, activation='linear', border_mode='same')(x)
x = UpSampling1D(length=2)(x)
x = Convolution1D(66, 25, activation='linear', border_mode='same')(x)

def vae_loss(input_motion, x):
    mse_loss = objectives.mse(input_motion, x)
    kl_loss = - 0.5 * K.mean(1 + z_log_std - K.square(z_mean) - K.exp(z_log_std))
    return mse_loss + kl_loss

vae = Model(input_motion, x)

nadam = Nadam(lr=0.000005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)

vae.compile(optimizer=nadam, loss=vae_loss)

"""
vae.fit(x_train, x_train,
        shuffle=True,
        nb_epoch=nb_epoch,
        batch_size=batch_size)
"""

#vae.save_weights('cmu_hdm05_vae_cnn.h5')
#vae.save_weights(sys.argv[2])

vae.load_weights('../models/cmu/keras/cmu_hdm05_vae_cnn_v0.h5')

datasets_cmu, std, mean = load_cmu_small(rng)

x_test = datasets_cmu[0][0]
x_test = x_test.swapaxes(1, 2)

x_predicted = vae.predict(x_test, batch_size=batch_size)
x_predicted = x_predicted.swapaxes(1, 2)

result = x_predicted * (std + 1e-10) + mean

new1 = result[25:26]
new2 = result[26:27]
new3 = result[0:1]

animation_plot([new1, new2, new3], interval=15.15)

#
#x_train = datasets_ft[0][0][:1900]
#x_train = x_train.reshape(x_train.shape[0], np.prod(x_train.shape[1:]))
#
#x_valid = datasets_ft[1][0][:610]
#x_valid = x_valid.reshape(x_valid.shape[0], np.prod(x_valid.shape[1:]))
#
#x_test = datasets_ft[2][0][:610]
#x_test = x_test.reshape(x_test.shape[0], np.prod(x_test.shape[1:]))
#
#y_train = datasets_ft[0][1][:1900]
#y_valid = datasets_ft[1][1][:610]
#y_test  = datasets_ft[2][1][:610]

# build a model to project inputs on the latent space
#encoder = Model(x, z_mean)

# (900, 50)
#x_ft_train = encoder.predict(x_train, batch_size=batch_size)
#x_ft_valid = encoder.predict(x_valid, batch_size=batch_size)
#x_ft_test  = encoder.predict(x_test, batch_size=batch_size)

#x_ft_train = np.concatenate([x_ft_train, x_ft_valid])
#y_train    = np.concatenate([y_train, y_valid])

#x_ft_input = Input(shape=(latent_dim,))
#x_ft = Dense(1000)(x_ft_input)
#x_ft = BatchNormalization()(x_ft)
#x_ft = Activation('relu')(x_ft)
#x_ft = Dropout(0.3)(x_ft)
#x_ft = Dense(1000)(x_ft)
#x_ft = BatchNormalization()(x_ft)
#x_ft = Activation('relu')(x_ft)
#x_ft = Dense(25, activation='softmax')(x_ft)
#
#model_ft = Model(x_ft_input, x_ft)
#model_ft.compile(optimizer='nadam', loss='categorical_crossentropy',
#                 metrics=['accuracy'])
#
#earlyStopping = EarlyStopping(monitor='val_acc', patience=50, verbose=0, mode='max')
#
#model_ft.fit(x_ft_train, y_train,
#             shuffle=True,
#             nb_epoch=500,
#             batch_size=batch_size,
#             validation_data=(x_ft_valid, y_valid),
#             callbacks=[earlyStopping])
#
#print model_ft.evaluate(x_ft_test, y_test, batch_size=10)
#model_ft.save_weights('classifier.h5')

## display a 2D plot of the digit classes in the latent space
##plt.figure(figsize=(6, 6))
##plt.scatter(x_train_encoded[:, 0], x_train_encoded[:, 1], c=y_train)
##plt.colorbar()
##plt.show()
#
#### build a digit generator that can sample from the learned distribution
###decoder_input = Input(shape=(latent_dim,))
###_h_decoded = decoder_h(decoder_input)
###_x_decoded_mean = decoder_mean(_h_decoded)
###generator = Model(decoder_input, _x_decoded_mean)
###
#### display a 2D manifold of the digits
###n = 15  # figure with 15x15 digits
###digit_size = 28
###figure = np.zeros((digit_size * n, digit_size * n))
#### we will sample n points within [-15, 15] standard deviations
###grid_x = np.linspace(-15, 15, n)
###grid_y = np.linspace(-15, 15, n)
###
###for i, yi in enumerate(grid_x):
###    for j, xi in enumerate(grid_y):
###        z_sample = np.array([[xi, yi]]) * epsilon_std
###        x_decoded = generator.predict(z_sample)
###        digit = x_decoded[0].reshape(digit_size, digit_size)
###        figure[i * digit_size: (i + 1) * digit_size,
###               j * digit_size: (j + 1) * digit_size] = digit
###
###plt.figure(figsize=(10, 10))
###plt.imshow(figure)
###plt.show()
