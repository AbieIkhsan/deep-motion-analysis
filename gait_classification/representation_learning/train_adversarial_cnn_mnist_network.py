import numpy as np
import theano
import theano.tensor as T

from nn.ActivationLayer import ActivationLayer
from nn.HiddenLayer import HiddenLayer
from nn.Conv2DLayer import Conv2DLayer
from nn.Pool2DLayer import Pool2DLayer
from nn.NoiseLayer import NoiseLayer
from nn.BatchNormLayer import BatchNormLayer
from nn.DropoutLayer import DropoutLayer
from nn.Network import Network, AutoEncodingNetwork, InverseNetwork
from nn.AdversarialAdamTrainer import AdversarialAdamTrainer
from nn.ReshapeLayer import ReshapeLayer
from utils import load_mnist

rng = np.random.RandomState(23455)

datasets = load_mnist(rng)

shared = lambda d: theano.shared(d, borrow=True)

train_set_x, train_set_y = map(shared, datasets[0])
valid_set_x, valid_set_y = map(shared, datasets[1])
test_set_x, test_set_y   = map(shared, datasets[2])

train_set_x = train_set_x.reshape((50000, 1, 28, 28))
valid_set_x = valid_set_x.reshape((10000, 1, 28, 28))
test_set_x  = test_set_x.reshape((10000, 1, 28, 28))

batchsize = 100

generatorNetwork = Network(
    HiddenLayer(rng, (100, 1000)),
    BatchNormLayer(rng, (100, 1000)),
    ActivationLayer(rng, f='ReLU'),

    HiddenLayer(rng, (1000, 64*7*7)),
    BatchNormLayer(rng, (1000, 64*7*7)),
    ActivationLayer(rng, f='ReLU'),
    
    ReshapeLayer(rng, (batchsize, 64, 7, 7)),

    InverseNetwork(Pool2DLayer(rng, (batchsize, 64, 14, 14))),
    DropoutLayer(rng, 0.25),
    Conv2DLayer(rng, (32, 64, 5, 5), (batchsize, 64, 14, 14)),
    ActivationLayer(rng, f='ReLU'),

    InverseNetwork(Pool2DLayer(rng, (batchsize, 32, 28, 28))),
    DropoutLayer(rng, 0.25),
    Conv2DLayer(rng, (1, 32, 5, 5), (batchsize, 32, 28, 28)),
    ActivationLayer(rng, f='ReLU'),
)

discriminatorNetwork = Network(
    DropoutLayer(rng, 0.25),
    Conv2DLayer(rng, (64, 1, 5, 5), (batchsize * 2, 1, 28, 28)),
    ActivationLayer(rng, f='PReLU'),
    Pool2DLayer(rng, (batchsize * 2, 64, 28, 28)),

    DropoutLayer(rng, 0.25),    
    Conv2DLayer(rng, (128, 64, 5, 5), (batchsize * 2, 64, 14, 14)),
    ActivationLayer(rng, f='PReLU'),
    Pool2DLayer(rng, (batchsize * 2, 128, 14, 14)),

    ReshapeLayer(rng, (batchsize * 2, 128*7*7)),
    DropoutLayer(rng, 0.25),    
    HiddenLayer(rng, (128*7*7, 1)),
    BatchNormLayer(rng, (128*7*7, 1)),

)

def generative_cost(disc_fake_out):
    return T.nnet.binary_crossentropy(disc_fake_out, np.ones(1, dtype=theano.config.floatX)).mean()

def discriminative_cost(disc_fake_out, disc_real_out):
    disc_cost = T.nnet.binary_crossentropy(disc_fake_out, np.zeros(1, dtype=theano.config.floatX)).mean()
    disc_cost += T.nnet.binary_crossentropy(disc_real_out, np.ones(1, dtype=theano.config.floatX)).mean()
    disc_cost /= np.float32(2.0)
    
    return disc_cost

trainer = AdversarialAdamTrainer(rng=rng, 
                                batchsize=batchsize, 
                                gen_cost=generative_cost, 
                                disc_cost=discriminative_cost,
                                epochs=25)

trainer.train(gen_network=generatorNetwork, 
                                disc_network=discriminatorNetwork, 
                                train_input=train_set_x,
                                filename=None)
