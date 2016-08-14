'''This script demonstrates how to build a variational autoencoder with Keras.
Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
'''
import numpy as np
import matplotlib.pyplot as plt 
import sys

from nn.AnimationPlotLines import animation_plot

np.random.seed(23455)

from keras.layers import *
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.models import Model
from keras.models import Sequential
from keras import backend as K
from keras import objectives
from keras.callbacks import EarlyStopping
from keras.optimizers import Nadam
from tools.utils import load_hdm05_easy, load_cmu, load_locomotion, load_cmu_small

rng = np.random.RandomState(23455)

datasets, std, mean = load_locomotion(rng)

x_train = datasets[0][0][:320]
x_train = x_train.swapaxes(1, 2)

I = np.arange(len(x_train))
rng.shuffle(I)
x_train = x_train[I]

batch_size = 40
latent_dim = 200
intermediate_dim = 1600
epsilon_std = 0.01
nb_epoch = 40

discriminator = Sequential()
discriminator.add(Convolution1D(64, 25, input_shape=(x_train.shape[1], x_train.shape[2]), border_mode='same'))
discriminator.add(ELU())
discriminator.add(MaxPooling1D(pool_length=2, stride=None))
discriminator.add(Dropout(0.2))
discriminator.add(Convolution1D(128, 25, border_mode='same'))
discriminator.add(ELU())
discriminator.add(MaxPooling1D(pool_length=2, stride=None))
discriminator.add(Dropout(0.2))
discriminator.add(Flatten())
discriminator.add(Dense(output_dim=1))
discriminator.add(BatchNormalization(mode=2))
discriminator.add(Activation('sigmoid'))


generator = Sequential()
generator.add(Dense(input_dim=latent_dim, output_dim=30*256))
generator.add(BatchNormalization())
generator.add(ELU())
generator.add(Dropout(0.2))
generator.add(Reshape((30, 256)))
generator.add(UpSampling1D(length=2))
generator.add(Convolution1D(256, 25, border_mode='same'))
generator.add(BatchNormalization())
generator.add(ELU())
generator.add(Dropout(0.2))
generator.add(UpSampling1D(length=2))
generator.add(Convolution1D(128, 25, border_mode='same'))
generator.add(ELU())
generator.add(Dropout(0.2))
generator.add(UpSampling1D(length=2))
generator.add(Convolution1D(66, 25, border_mode='same'))
generator.add(ELU())

generator_discriminator = Sequential()
generator_discriminator.add(generator)
discriminator.trainable = False
generator_discriminator.add(discriminator)

nadam = Nadam(lr=0.000001, beta_1=0.7, beta_2=0.999, epsilon=1e-08)

generator.compile(optimizer=nadam, loss='binary_crossentropy')
generator_discriminator.compile(optimizer=nadam, loss='binary_crossentropy')
discriminator.trainable = True
discriminator.compile(optimizer=nadam, loss='binary_crossentropy')

for epoch in range(nb_epoch):
    train_batchinds = np.arange(x_train.shape[0] // batch_size)
    np.random.shuffle(train_batchinds)

    for ii, index in enumerate(train_batchinds):
        batched_input = x_train[index*batch_size:(index+1)*batch_size]

        random_prior = np.random.uniform(size=(batch_size, latent_dim), low=-2, high=2)
        fake_motions = generator.predict(random_prior)
        X = np.concatenate([batched_input, fake_motions], axis=0)

        y = [1] * batch_size + [0] * batch_size
        disc_cost = discriminator.train_on_batch(X, y)

        for i in xrange(2):
            random_prior = np.random.uniform(size=(batch_size, latent_dim), low=-2, high=2)
            gen_cost = generator_discriminator.train_on_batch(random_prior, [1] * batch_size)

            sys.stdout.write('\r[Epoch %i]   generative loss: %.5f   discriminative loss: %.5f' % (epoch, gen_cost, disc_cost))
            sys.stdout.flush()

        if ii % 10 == 9:
            generator.save_weights('../models/cmu/keras/adv/generator_v0.hd5', True)
            discriminator.save_weights('../models/cmu/keras/adv/discriminator_v0.hd5', True)

generator.load_weights('../models/cmu/keras/adv/generator_v0.hd5')

random_prior = np.random.uniform(size=(100, latent_dim), low=-2, high=2)

x_predicted = generator.predict(random_prior, batch_size=100)
x_predicted = x_predicted.swapaxes(1, 2)

result = x_predicted * (std + 1e-10) + mean

new1 = result[25:26]
new2 = result[26:27]
new3 = result[0:1]

animation_plot([new1, new2, new3], interval=15.15)