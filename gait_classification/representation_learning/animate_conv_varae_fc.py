import numpy as np
import theano
import theano.tensor as T

from nn.AdamTrainer import AdamTrainer
from nn.ActivationLayer import ActivationLayer
from nn.AnimationPlotLines import animation_plot
from nn.DropoutLayer import DropoutLayer
from nn.Pool1DLayer import Pool1DLayer
from nn.Conv1DLayer import Conv1DLayer
from nn.ReshapeLayer import ReshapeLayer
from nn.HiddenLayer import HiddenLayer
from nn.VariationalLayer import VariationalLayer
from nn.Network import Network, AutoEncodingNetwork, InverseNetwork
from nn.AnimationPlotLines import animation_plot, plot_movement

from tools.utils import load_cmu, load_cmu_small, load_locomotion

rng = np.random.RandomState(23455)

dataset, std, mean = load_locomotion(rng)

BATCH_SIZE = 50

generatorNetwork = Network(
    DropoutLayer(rng, 0.25),  
    HiddenLayer(rng, (600, 64*60)),
    ActivationLayer(rng, f='elu'),
    ReshapeLayer(rng, (BATCH_SIZE, 64, 60)),

    InverseNetwork(Pool1DLayer(rng, (2,), (BATCH_SIZE, 64, 120))),
    DropoutLayer(rng, 0.25),    
    Conv1DLayer(rng, (64, 64, 25), (BATCH_SIZE, 64, 120)),
    ActivationLayer(rng, f='elu'),

    InverseNetwork(Pool1DLayer(rng, (2,), (BATCH_SIZE, 64, 240))),
    DropoutLayer(rng, 0.25),    
    Conv1DLayer(rng, (66, 64, 25), (BATCH_SIZE, 64, 240)),
    ActivationLayer(rng, f='elu'),
)

generatorNetwork.load([None, '../models/cmu/conv_varae/v_9/layer_3.npz', None, None,
                        None, None, '../models/cmu/conv_varae/v_9/layer_4.npz', None,
                        None, None, '../models/cmu/conv_varae/v_9/layer_5.npz', None])

def randomize_uniform_data(n_input):
    return rng.uniform(size=(n_input, 600), 
            low=-np.sqrt(5, dtype=theano.config.floatX), 
            high=np.sqrt(5, dtype=theano.config.floatX)).astype(theano.config.floatX)

gen_rand_input = theano.shared(randomize_uniform_data(50), name = 'z')
generate_sample_motions = theano.function([], generatorNetwork(gen_rand_input))
sample = generate_sample_motions()

result = sample * (std + 1e-10) + mean

new1 = result[25:26]
new2 = result[26:27]
new3 = result[0:1]

animation_plot([new1, new2, new3], filename='vae-locomotion-750.mp4', interval=15.15)

new1 = new1.reshape((66,240))

#np.savez_compressed('cmu_mean_std.npz', mean=mean, std=std)
#np.savez_compressed('vae_gan_cmu_50_result.npz', result=result)
#animation_plot([new1, new2, new3], filename= 'vae-gan.mp4', interval=15.15)

for i in xrange(240):
    plot_movement(new1, filename='vae_locomotion/vae_locomotion'+str(i)+'.png', n_figs=i)
