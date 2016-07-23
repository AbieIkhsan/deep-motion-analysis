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
from nn.Network import Network
from nn.AdversarialAeAdamTrainer import AdversarialAdamTrainer
from nn.ReshapeLayer import ReshapeLayer

from tools.utils import load_mnist

rng = np.random.RandomState(23455)

datasets = load_mnist(rng)

shared = lambda d: theano.shared(d, borrow=True)

train_set_x, train_set_y = map(shared, datasets[0])
valid_set_x, valid_set_y = map(shared, datasets[1])
test_set_x, test_set_y   = map(shared, datasets[2])

batchsize = 100

encoderNetwork = Network(
	HiddenLayer(rng, (784, 1200)),
	BatchNormLayer(rng, (784, 1200)),
	ActivationLayer(rng, f='ReLU'),
	HiddenLayer(rng, (1200, 1200)),
	#BatchNormLayer(rng, (1200, 1200)),
	ActivationLayer(rng, f='ReLU'),
	HiddenLayer(rng, (1200, 1200)),
	#BatchNormLayer(rng, (1200, 1200)),
	ActivationLayer(rng, f='ReLU'),
	HiddenLayer(rng, (1200, 100)),
)

decoderNetwork = Network(
	HiddenLayer(rng, (100, 240)),
	BatchNormLayer(rng, (100, 240)),
	ActivationLayer(rng, f='ReLU'),
	DropoutLayer(rng, 0.2),
	HiddenLayer(rng, (240, 1200)),
	BatchNormLayer(rng, (240, 1200)),
	ActivationLayer(rng, f='ReLU'),
	DropoutLayer(rng, 0.2),
	HiddenLayer(rng, (1200, 1200)),
	#BatchNormLayer(rng, (1200, 1200)),
	ActivationLayer(rng, f='ReLU'),
	DropoutLayer(rng, 0.2),
	HiddenLayer(rng, (1200, 784)),
	#BatchNormLayer(rng, (1200, 784)),
	ActivationLayer(rng, f='sigmoid'),	
)

discriminatorNetwork = Network(
	HiddenLayer(rng, (100, 240)),
	BatchNormLayer(rng, (100, 240)),
	ActivationLayer(rng, f='ReLU'),
	DropoutLayer(rng, 0.2),
	HiddenLayer(rng, (240, 1200)),
	#BatchNormLayer(rng, (240, 1200)),
	ActivationLayer(rng, f='ReLU'),
	DropoutLayer(rng, 0.2),
	HiddenLayer(rng, (1200, 1200)),
	#BatchNormLayer(rng, (1200, 1200)),
	ActivationLayer(rng, f='ReLU'),
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

    return reg_weight * reg_cost + recons_weight * recons_cost

def decoder_cost(dec_out, input_data):
	return T.sqr(dec_out - input_data).mean()

def discriminative_cost(disc_fake_out, disc_real_out):
	disc_cost = T.nnet.binary_crossentropy(disc_real_out, np.zeros(1, dtype=theano.config.floatX)).mean()
	disc_cost += T.nnet.binary_crossentropy(disc_fake_out, np.ones(1, dtype=theano.config.floatX)).mean()
	disc_cost /= np.float32(2.0)
	
	return disc_cost

trainer = AdversarialAdamTrainer(rng=rng, 
								batchsize=batchsize, 
								enc_cost=encoder_cost, 
								dec_cost=decoder_cost,
								disc_cost=discriminative_cost,
								epochs=20)

trainer.train(enc_network=encoderNetwork, 
								dec_network=decoderNetwork,
								disc_network=discriminatorNetwork, 
								train_input=train_set_x,
                                filename=None)
