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
from nn.AdversarialAeAdamTrainer import AdversarialAdamTrainer
from nn.ReshapeLayer import ReshapeLayer

from nn.AnimationPlotLines import animation_plot

from tools.utils import load_cmu, load_cmu_small, load_locomotion

rng = np.random.RandomState(23455)

shared = lambda d: theano.shared(d, borrow=True)

dataset, std, mean = load_locomotion(rng)
E = shared(dataset[0][0])

BATCH_SIZE = 32

encoderNetwork = Network(
    DropoutLayer(rng, 0.25),
    Conv1DLayer(rng, (64, 66, 25), (BATCH_SIZE, 66, 240)),
    BatchNormLayer(rng, (BATCH_SIZE, 64, 240)),
    Pool1DLayer(rng, (2,), (BATCH_SIZE, 64, 240)),
    ActivationLayer(rng, f='elu'),

    DropoutLayer(rng, 0.25),
    Conv1DLayer(rng, (128, 64, 25), (BATCH_SIZE, 64, 120)),
    BatchNormLayer(rng, (BATCH_SIZE, 128, 120)),
    Pool1DLayer(rng, (2,), (BATCH_SIZE, 128, 120)),
    ActivationLayer(rng, f='elu'),

    ReshapeLayer(rng, (BATCH_SIZE, 128*60)),
    DropoutLayer(rng, 0.25),    
    HiddenLayer(rng, (128*60, 800)),
    ActivationLayer(rng, f='elu'),
)

decoderNetwork = Network(
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

discriminatorNetwork = Network(
    HiddenLayer(rng, (800, 1200)),
    BatchNormLayer(rng, (800, 1200)),
    ActivationLayer(rng, f='elu'),
    DropoutLayer(rng, 0.2),
    HiddenLayer(rng, (1200, 1200)),
    BatchNormLayer(rng, (240, 1200)),
    ActivationLayer(rng, f='elu'),
    DropoutLayer(rng, 0.2),
    HiddenLayer(rng, (1200, 1200)),
    #BatchNormLayer(rng, (1200, 1200)),
    ActivationLayer(rng, f='elu'),
    DropoutLayer(rng, 0.2),
    HiddenLayer(rng, (1200, 1)),
    #BatchNormLayer(rng, (1200, 1)),
    ActivationLayer(rng, f='sigmoid'),  
)

def encoder_cost(disc_out, dec_out, input_data):
    reg_cost = T.nnet.binary_crossentropy(disc_out, np.ones(1, dtype=theano.config.floatX)).mean()
    recons_cost = T.sqr(dec_out - input_data).mean()

    reg_weight = 50
    recons_weight = 1

    return reg_weight * reg_cost + recons_weight * recons_cost, recons_cost

def decoder_cost(dec_out, input_data):
    return T.sqr(dec_out - input_data).mean()

def discriminative_cost(disc_fake_out, disc_real_out):
    disc_cost = T.nnet.binary_crossentropy(disc_real_out, np.zeros(1, dtype=theano.config.floatX)).mean()
    disc_cost += T.nnet.binary_crossentropy(disc_fake_out, np.ones(1, dtype=theano.config.floatX)).mean()
    disc_cost /= np.float32(2.0)
    
    return disc_cost

trainer = AdversarialAdamTrainer(rng=rng, 
                                    batchsize=BATCH_SIZE, 
                                    enc_cost=encoder_cost, 
                                    dec_cost=decoder_cost,
                                    disc_cost=discriminative_cost,
                                    epochs=750)

trainer.train(enc_network=encoderNetwork, 
                                dec_network=decoderNetwork,
                                disc_network=discriminatorNetwork, 
                                train_input=E,
                                filename=['../models/locomotion/adv_ae/v_3/layer_0.npz',
                                        '../models/locomotion/adv_ae/v_3/layer_1.npz', 
                                        None, None,
                                        None, None, '../models/locomotion/adv_ae/v_3/layer_2.npz', None,
                                        None, None, '../models/locomotion/adv_ae/v_3/layer_3.npz', None,
                                        None, None, '../models/locomotion/adv_ae/v_3/layer_4.npz', None,])


BATCH_SIZE = 50

generatorNetwork = Network(
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

generatorNetwork.load(['../models/locomotion/adv_ae/v_3/layer_0.npz',
                                        '../models/locomotion/adv_ae/v_3/layer_1.npz', 
                                        None, None,
                                        None, None, '../models/locomotion/adv_ae/v_3/layer_2.npz', None,
                                        None, None, '../models/locomotion/adv_ae/v_3/layer_3.npz', None,
                                        None, None, '../models/locomotion/adv_ae/v_3/layer_4.npz', None,])

def randomize_uniform_data(n_input):
    return rng.uniform(size=(n_input, 800), 
            low=-np.sqrt(5, dtype=theano.config.floatX), 
            high=np.sqrt(5, dtype=theano.config.floatX)).astype(theano.config.floatX)

gen_rand_input = theano.shared(randomize_uniform_data(50), name = 'z')
generate_sample_motions = theano.function([], generatorNetwork(gen_rand_input))
sample = generate_sample_motions()

result = sample * (std + 1e-10) + mean

new1 = result[25:26]
new2 = result[26:27]
new3 = result[0:1]

animation_plot([new1, new2, new3], filename= 'ae-gan-locomotion.mp4', interval=15.15)