import matplotlib.pyplot as plt
import numpy as np

import theano
import theano.tensor as T

from nn.ActivationLayer import ActivationLayer
from nn.HiddenLayer import HiddenLayer
from nn.Conv1DLayer import Conv1DLayer
from nn.Pool1DLayer import Pool1DLayer
from nn.NoiseLayer import NoiseLayer
from nn.BatchNormLayer import BatchNormLayer
from nn.DropoutLayer import DropoutLayer
from nn.Network import Network, AutoEncodingNetwork, InverseNetwork
from nn.AdversarialAdamTrainer import AdversarialAdamTrainer
from nn.ReshapeLayer import ReshapeLayer
from nn.AnimationPlotLines import animation_plot, plot_movement
from tools.utils import load_cmu, load_cmu_small, load_locomotion

rng = np.random.RandomState(23455)

shared = lambda d: theano.shared(d, borrow=True)

dataset, std, mean = load_locomotion(rng)

BATCH_SIZE = 1000

generatorNetwork = Network(
    DropoutLayer(rng, 0.25),    
    HiddenLayer(rng, (800, 64*30)),
    BatchNormLayer(rng, (800, 64*30)),
    ActivationLayer(rng, f='elu'),
    ReshapeLayer(rng, (BATCH_SIZE, 64, 30)),

    InverseNetwork(Pool1DLayer(rng, (2,), (BATCH_SIZE, 64, 60))),
    DropoutLayer(rng, 0.15),    
    Conv1DLayer(rng, (64, 64, 25), (BATCH_SIZE, 64, 60)),
    ActivationLayer(rng, f='elu'),

    InverseNetwork(Pool1DLayer(rng, (2,), (BATCH_SIZE, 64, 120))),
    DropoutLayer(rng, 0.25),  
    Conv1DLayer(rng, (64, 64, 25), (BATCH_SIZE, 64, 120)),
    ActivationLayer(rng, f='elu'),

    InverseNetwork(Pool1DLayer(rng, (2,), (BATCH_SIZE, 64, 240))),
    DropoutLayer(rng, 0.25),    
    Conv1DLayer(rng, (66, 64, 25), (BATCH_SIZE, 64, 240)),
    ActivationLayer(rng, f='elu'),
)

generatorNetwork.load([None, '../models/locomotion/adv/v_2/layer_0.npz',
                                        '../models/locomotion/adv/v_2/layer_1.npz', 
                                        None, None,
                                        None, None, '../models/locomotion/adv/v_2/layer_2.npz', None,
                                        None, None, '../models/locomotion/adv/v_2/layer_3.npz', None,
                                        None, None, '../models/locomotion/adv/v_2/layer_4.npz', None,])

def randomize_uniform_data(n_input):
    return rng.uniform(size=(n_input, 800), 
            low=-3, 
            high=3).astype(theano.config.floatX)

gen_rand_input = theano.shared(randomize_uniform_data(1000), name = 'z')
generate_sample_motions = theano.function([], generatorNetwork(gen_rand_input))
sample = generate_sample_motions()

result = sample * (std + 1e-10) + mean

new1 = result[118:119]
new2 = result[616:617]
new3 = result[870:871]

animation_plot([new1, new2, new3], interval=15.15)

new1 = new1.reshape((66,240))

for i in xrange(240):
    plot_movement(new1, filename='gan_locomotion/gan_locomotion'+str(i)+'.png', n_figs=i)