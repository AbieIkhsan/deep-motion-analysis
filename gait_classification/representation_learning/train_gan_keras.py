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

datasets, std, mean = load_cmu(rng)

x_train = datasets[0][0][:12000].swapaxes(1, 2)
x_valid = datasets[0][0][12000:16000].swapaxes(1, 2)

#x_train = np.concatenate([x_train, noisy])
#I = np.arange(len(x_train))
#rng.shuffle(I)
#x_train = x_train[I]

batch_size = 10
latent_dim = 100
nb_epoch   = 20

#input_motion = Input(batch_shape=(batch_size, x_train.shape[1], x_train.shape[2]))

def discriminator_network():
    model = Sequential()
    model.add(Convolution1D(64, 25, input_shape=(x_train.shape[1], x_train.shape[2]), border_mode='same'))
    model.add(BatchNormalization(mode=2))
    model.add(LeakyReLU(0.2))
    model.add(MaxPooling1D(pool_length=2, stride=None))
    model.add(Convolution1D(128, 25, border_mode='same'))
    model.add(BatchNormalization(mode=2))
    model.add(LeakyReLU(0.2))
    model.add(MaxPooling1D(pool_length=2, stride=None))
    model.add(Convolution1D(256, 25, border_mode='same'))
    model.add(BatchNormalization(mode=2))
    model.add(LeakyReLU(0.2))
    model.add(MaxPooling1D(pool_length=2, stride=None))
    model.add(Flatten())
    model.add(Dense(output_dim=1))
    model.add(Activation('sigmoid'))

    return model


def generator_network():
    model = Sequential()
    model.add(Dense(input_dim=latent_dim, output_dim=30*256))
    #model.add(BatchNormalization())
    model.add(BatchNormalization(mode=2))
    model.add(ELU())
    model.add(Dropout(0.4))
    model.add(Reshape((30, 256)))
    model.add(UpSampling1D(length=2))
    model.add(Convolution1D(256, 25, border_mode='same'))
    #model.add(BatchNormalization())
    model.add(BatchNormalization(mode=2))
    model.add(ELU())
    model.add(Dropout(0.4))
    model.add(UpSampling1D(length=2))
    model.add(Convolution1D(128, 25, border_mode='same'))
    model.add(BatchNormalization(mode=2))
    #model.add(BatchNormalization())
    model.add(ELU())
    #model.add(Dropout(0.4))
    model.add(UpSampling1D(length=2))
    model.add(Convolution1D(66, 25, border_mode='same'))

    return model

def generator_discriminator_network(generator, discriminator):
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)

    return model



discriminator           = discriminator_network()
generator               = generator_network()
generator_discriminator = generator_discriminator_network(generator, discriminator)

nadam = Nadam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

#generator.compile(optimizer=nadam, loss='binary_crossentropy')
#discriminator.trainable = True

generator.compile(optimizer=nadam, loss='binary_crossentropy')
discriminator.compile(optimizer=nadam, loss='binary_crossentropy', metrics=['accuracy'])
generator_discriminator.compile(optimizer=nadam, loss='binary_crossentropy', metrics=['accuracy'])

y_train = np.concatenate([np.ones([x_train.shape[0]]), np.zeros([x_train.shape[0]])])
y_valid = np.concatenate([np.ones([x_valid.shape[0]]), np.zeros([x_valid.shape[0]])])

for i in range(50):
    generator_input       = np.random.uniform(size=(x_train.shape[0], latent_dim))
    generator_valid_input = np.random.uniform(size=(x_valid.shape[0], latent_dim))

    train_noisy = generator.predict(generator_input, batch_size=batch_size)
    valid_noisy = generator.predict(generator_valid_input, batch_size=batch_size)

    d_input = np.concatenate([x_train, train_noisy])
    d_v_input = np.concatenate([x_valid, valid_noisy])

    discriminator.fit(d_input, y_train,
                      shuffle=True,
                      nb_epoch=nb_epoch,
                      batch_size=batch_size,
                      validation_data=(d_v_input, y_valid))

    print '\n\nGenerator:\n'

    generator_discriminator.fit(generator_input, np.ones(generator_input.shape[0]),
                                shuffle=True,
                                nb_epoch=nb_epoch,
                                batch_size=batch_size,
                                validation_data=(generator_valid_input, np.ones(generator_valid_input.shape[0])))

"""
#for epoch in range(nb_epoch):
#    train_batchinds = np.arange(x_train.shape[0] // batch_size)
#    np.random.shuffle(train_batchinds)
#
#    for ii, index in enumerate(train_batchinds):
#        batched_input = x_train[index*batch_size:(index+1)*batch_size]

#        random_prior = np.random.uniform(size=(batch_size, 800), low=-np.sqrt(3), high=np.sqrt(3))
#        fake_motions = generator.predict(random_prior)
#        X = np.concatenate([batched_input, fake_motions], axis=0)

#        y = [1] * batch_size + [0] * batch_size
#        disc_cost = discriminator.train_on_batch(X, y)

        for i in xrange(2):
            random_prior = np.random.uniform(size=(batch_size, 800), low=-np.sqrt(3), high=np.sqrt(3))
            #gen_cost = generator_discriminator.train_on_batch(random_prior, [1] * batch_size)
            gen_cost = np.inf

            sys.stdout.write('[Epoch %i]   generative cost: %.5f   discriminative cost: %.5f\n' % (epoch, gen_cost, disc_cost))
            sys.stdout.flush()

        if ii % 10 == 9:
            generator.save_weights('../models/cmu/keras/adv/generator_v0.hd5', True)
            discriminator.save_weights('../models/cmu/keras/adv/discriminator_v0.hd5', True)


generator.load_weights('../models/cmu/keras/adv/generator_v0.hd5')

random_prior = np.random.uniform(size=(100, 800), low=-np.sqrt(3), high=np.sqrt(3))
"""

x_predicted = generator.predict(generator_input[:10], batch_size=batch_size)

x_predicted = x_predicted.swapaxes(1, 2)

result = x_predicted * (std + 1e-10) + mean

new1 = result[2:3]
new2 = result[1:2]
new3 = result[0:1]

animation_plot([new1, new2, new3], interval=15.15)
