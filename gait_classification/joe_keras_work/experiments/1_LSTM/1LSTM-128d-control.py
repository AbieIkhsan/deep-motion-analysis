'''
    Train on t+1 vector for every t, only trying to predict the final frame, given 29 as seed.
'''

from __future__ import print_function

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, TimeDistributed
from keras.layers import LSTM, GRU
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Nadam
import keras
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import theano
sys.path.append('../../../representation_learning/')

from nn.Network import InverseNetwork, AutoEncodingNetwork
from nn.AnimationPlot import animation_plot


data = np.load('../../../data/Joe/sequential_final_frame.npz')
train_x = data['train_x']
train_y = data['train_y']
test_x = data['test_x']
test_y = data['test_y']

control = np.load('../../../data/Joe/edin_shuffled_control.npz')['control']
control = control.swapaxes(1,2)
train_control = control[:310,::8]
train_n = np.concatenate((train_x, train_control[:,:29]), axis=2)
train_m = np.concatenate((train_y, train_control[:,1:]), axis=2)

test_control = control[310:,::8]
test_n = np.concatenate((test_x, test_control[:,:29]), axis=2)
test_m = np.concatenate((test_y, test_control[:,1:]), axis=2)

print(train_n.shape)
# build the model: 2 stacked LSTM
print('Build model...')
model = Sequential()
model.add(TimeDistributed(Dense(128), input_shape=(29, 259)))
model.add(Activation(keras.layers.advanced_activations.ELU(alpha=1.0)))
model.add(LSTM(256, return_sequences=True, consume_less='gpu', \
               init='glorot_normal'))
model.add(TimeDistributed(Dense(259)))
model.add(Activation(keras.layers.advanced_activations.ELU(alpha=1.0)))
# TimedistributedDense on top - Can then set output vectors to be next sequence!

model.compile(loss='mean_squared_error', optimizer='nadam')

print('Training model...')
model.fit(train_n, train_m, batch_size=10, nb_epoch=200, validation_data=(test_n,test_m))

score = model.evaluate(test_n,test_m)
print(score)
model.save_weights('../../weights/1LSTM-128d-first.hd5', overwrite=True)

train_control = np.zeros((310,30,3))

for i in range(30):
    train_control[:,i] = np.mean(control[:310,8*i:(i+1)*8], axis=1)
    
train_n = np.concatenate((train_x, train_control[:,:29]), axis=2)
train_m = np.concatenate((train_y, train_control[:,1:]), axis=2)

for i in range(30):
    test_control[:,i] = np.mean(control[310:,i*8:(i+1)*8], axis=1)

test_n = np.concatenate((test_x, test_control[:,:29]), axis=2)
test_m = np.concatenate((test_y, test_control[:,1:]), axis=2)

# build the model: 2 stacked LSTM
print('Build model...')
model = Sequential()
model.add(TimeDistributed(Dense(128), input_shape=(29, 259)))
model.add(Activation(keras.layers.advanced_activations.ELU(alpha=1.0)))
model.add(LSTM(256, return_sequences=True, consume_less='gpu', \
               init='glorot_normal'))
model.add(TimeDistributed(Dense(259)))
model.add(Activation(keras.layers.advanced_activations.ELU(alpha=1.0)))
# TimedistributedDense on top - Can then set output vectors to be next sequence!

model.compile(loss='mean_squared_error', optimizer='nadam')

print('Training model...')
model.fit(train_n, train_m, batch_size=10, nb_epoch=200, validation_data=(test_n,test_m))

score = model.evaluate(test_n,test_m)
print(score)
model.save_weights('../../weights/1LSTM-128d-mean.hd5', overwrite=True)
