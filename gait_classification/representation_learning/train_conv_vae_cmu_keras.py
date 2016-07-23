import numpy as np
import matplotlib.pyplot as plt

from keras.layers import *
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.models import Model
from tools.utils import load_cmu
from tools.utils import load_styletransfer
from nn.AnimationPlotLines import animation_plot

rng = np.random.RandomState(23455)

datasets, std, mean = load_styletransfer(rng, 80)

print len(datasets)

# Shape = (MB, 66, 240)
x_train = datasets[0][0].swapaxes(1, 2)
x_test = datasets[0][0].swapaxes(1, 2)
# Shape = (MB, 240, 66)
input_motion = Input(shape=x_train.shape[1:])

encoder_0 = Convolution1D(64, 25, activation='relu', border_mode='same')
encoder_1 = MaxPooling1D(pool_length=2, stride=None)
encoder_2 = Convolution1D(128, 25, border_mode='same')
encoder_3 = MaxPooling1D(pool_length=2, stride=None)
encoder_4 = Convolution1D(256, 25, border_mode='same')
encoder_5 = MaxPooling1D(pool_length=2, stride=None)

x = encoder_0(input_motion)
x = encoder_1(x)
x = encoder_2(x)
x = encoder_3(x)
x = encoder_4(x)
x = encoder_5(x)

# at this point the representation is (256, 30)
decoder_0 = Flatten()
decoder_1 = Dense(50)
decoder_2 = Dense(30*66)
decoder_3 = Reshape((30, 66))
decoder_4 = UpSampling1D(length=2)
decoder_5 = Convolution1D(256, 25, border_mode='same')
decoder_6 = UpSampling1D(length=2)
decoder_7 = Convolution1D(256, 25, border_mode='same')
decoder_8 = UpSampling1D(length=2)
decoder_9 = Convolution1D(66, 25, activation='linear', border_mode='same')

x = decoder_0(x)
x = decoder_1(x)
x = decoder_2(x)
x = decoder_3(x)
x = decoder_4(x)
x = decoder_5(x)
x = decoder_6(x)
x = decoder_7(x)
x = decoder_8(x)
decoded = decoder_9(x)

autoencoder = Model(input_motion, decoded)
autoencoder.compile(optimizer='adadelta', loss='mse')

autoencoder.fit(x_train, x_train,
                nb_epoch=5,
                batch_size=100,
                shuffle=True)

reconst_motion = autoencoder.predict(x_test)
result = reconst_motion.swapaxes(1, 2)*(std + 1e-10) + mean

new1 = result[250:251]
new2 = result[269:270]
new3 = result[0:1]

animation_plot([new1, new2, new3], interval=15.15)

"""
# build a digit generator that can sample from the learned distribution
decoder_input = Input(shape=(50,))
h = decoder_0(decoder_input)
h = decoder_1(h)
h = decoder_2(h)
h = decoder_3(h)
h = decoder_4(h)
h = decoder_5(h)
h = decoder_6(h)
h = decoder_7(h)
decoded = decoder_8(x)

generator = Model(decoder_input, _x_decoded_mean)

result = generator.predict

* (std + 1e-10) + mean

new1 = result[250:251]
new2 = result[269:270]
new3 = result[0:1]

animation_plot([new1, new2, new3], interval=15.15)
"""