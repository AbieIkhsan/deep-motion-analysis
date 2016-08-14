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
from nn.VariationalLayer import VariationalLayer
from nn.Network import Network, AutoEncodingNetwork, InverseNetwork
from nn.AdversarialVaeAdamTrainer import AdversarialAdamTrainer
from nn.ReshapeLayer import ReshapeLayer

from nn.AnimationPlotLines import animation_plot, plot_movement

from tools.utils import load_cmu, load_cmu_small, load_locomotion

rng = np.random.RandomState(23455)

shared = lambda d: theano.shared(d, borrow=True)

dataset, std, mean = load_cmu_small(rng)
E = shared(dataset[0][0])

BATCH_SIZE = 100

FC_SIZE = 800

generatorNetwork = Network(
    HiddenLayer(rng, (FC_SIZE/2, 64*30)),
    BatchNormLayer(rng, (FC_SIZE/2, 64*30)),
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

generatorNetwork.load(['../models/locomotion/adv_vae/v_2/layer_0.npz',
                                        '../models/locomotion/adv_vae/v_2/layer_1.npz', 
                                        None, None,
                                        None, None, '../models/locomotion/adv_vae/v_2/layer_2.npz', None,
                                        None, None, '../models/locomotion/adv_vae/v_2/layer_3.npz', None,
                                        None, None, '../models/locomotion/adv_vae/v_2/layer_4.npz', None,])

def randomize_uniform_data(n_input):
    return rng.uniform(size=(n_input, FC_SIZE/2), 
            low=-np.sqrt(3, dtype=theano.config.floatX), 
            high=np.sqrt(3, dtype=theano.config.floatX)).astype(theano.config.floatX)

gen_rand_input = theano.shared(randomize_uniform_data(100), name = 'z')
generate_sample_motions = theano.function([], generatorNetwork(gen_rand_input))
sample = generate_sample_motions()

result = sample * (std + 1e-10) + mean

new1 = result[35:36]
new2 = result[26:27]
new3 = result[0:1]

animation_plot([new1, new2, new3], filename= 'vae-gan-2.mp4', interval=15.15)

new1 = new1.reshape((66,240))

for i in xrange(240):
    plot_movement(new1, filename='vae_gan_locomotion/vae_gan_locomotion'+str(i)+'.png', n_figs=i)

#plot_movement(new1, 0)
#plot_movement(new1, 96)
#plot_movement(new1, n_figs=25)
#plot_movement(new1, 144)
#plot_movement(new1, 192)