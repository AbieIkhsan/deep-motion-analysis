import numpy as np
import theano
import theano.tensor as T

from nn.ActivationLayer import ActivationLayer
from nn.BatchNormLayer import BatchNormLayer
from nn.HiddenLayer import HiddenLayer
from nn.Network import Network
from nn.AdamTrainer import AdamTrainer
from nn.VariationalLayer import VariationalLayer

import matplotlib.pyplot as plt

from tools.utils import load_mnist

rng = np.random.RandomState(23455)

datasets = load_mnist(rng)

shared = lambda d: theano.shared(d, borrow=True)

train_set_x, train_set_y = map(shared, datasets[0])
valid_set_x, valid_set_y = map(shared, datasets[1])
test_set_x, test_set_y   = map(shared, datasets[2])

network = Network(
	Network(
	    HiddenLayer(rng, (784, 256)),
        #BatchNormLayer(rng, (784, 128)),
	    ActivationLayer(rng, f='elu'),

	    HiddenLayer(rng, (256, 64)),
        #BatchNormLayer(rng, (256, 4)),
	    ActivationLayer(rng, f='elu'),
	),

	Network(
        VariationalLayer(rng),
    ),

	Network(
	    HiddenLayer(rng, (32, 64)),
        #BatchNormLayer(rng, (2, 64)),
    	ActivationLayer(rng, f='elu'),
    	HiddenLayer(rng, (64, 256)),
        #BatchNormLayer(rng, (64, 256)),
    	ActivationLayer(rng, f='elu'),
    	HiddenLayer(rng, (256, 784)),
        #BatchNormLayer(rng, (256, 784)),
    	ActivationLayer(rng, f='sigmoid'),
    )
)

def cost(networks, X, Y):
    network_u, network_v, network_d = networks.layers
    
    vari_amount = 1.0
    repr_amount = 1.0
    
    H = network_u(X)
    mu, sg = H[:,0::2], H[:,1::2]

    # ya0st VAE
    vari_cost = 0.5 * T.sum(1 + 2 * sg - mu**2 - T.exp(2 * sg))
    repr_cost = T.sum((network_d(network_v(H)) - Y)**2)

    # previous VAE
    #vari_cost = 0.5 * T.mean(mu**2) + 0.5 * T.mean((T.sqrt(T.exp(sg))-1)**2)
    #repr_cost = T.mean((network_d(network_v(H)) - Y)**2)
    

    #return repr_amount * repr_cost + vari_amount * vari_cost
    
    #return T.mean(repr_cost - vari_cost)
    return repr_amount * repr_cost - vari_amount * vari_cost

trainer = AdamTrainer(rng=rng, batchsize=128, epochs=50, alpha=0.0001, cost=cost)
trainer.train(network=network, train_input=train_set_x, train_output=train_set_x, 
    filename=[['../models/mnist/mlp_varae/v_0/layer_0.npz', 
                                        #'../models/mnist/mlp_varae/v_0/layer_5.npz', 
                                        None, 
                                        '../models/mnist/mlp_varae/v_0/layer_1.npz', 
                                        None, ],
                                        [None,],
                                        ['../models/mnist/mlp_varae/v_0/layer_2.npz', 
                                        #'../models/mnist/mlp_varae/v_0/layer_6.npz', 
                                        None,
                                        '../models/mnist/mlp_varae/v_0/layer_3.npz', 
                                        None,
                                        '../models/mnist/mlp_varae/v_0/layer_4.npz', 
                                        None,],])

def randomize_uniform_data(n_input):
    return rng.uniform(size=(n_input, 32), 
            low=-np.sqrt(10, dtype=theano.config.floatX), 
            high=np.sqrt(10, dtype=theano.config.floatX)).astype(theano.config.floatX)

testNetwork = Network(
        HiddenLayer(rng, (32, 64)),
        #BatchNormLayer(rng, (2, 64)),
        ActivationLayer(rng, f='elu'),
        HiddenLayer(rng, (64, 256)),
        #BatchNormLayer(rng, (64, 256)),
        ActivationLayer(rng, f='elu'),
        HiddenLayer(rng, (256, 784)),
        ActivationLayer(rng, f='sigmoid'),
    )

testNetwork.load(['../models/mnist/mlp_varae/v_0/layer_2.npz', 
                        None,
                        '../models/mnist/mlp_varae/v_0/layer_3.npz', 
                        None,
                        '../models/mnist/mlp_varae/v_0/layer_4.npz', 
                        None,])

gen_rand_input = theano.shared(randomize_uniform_data(100), name = 'z')
generate_sample_images = theano.function([], testNetwork(gen_rand_input))
sample = generate_sample_images()

sample = sample.reshape((10,10,28,28)).transpose(1,2,0,3).reshape((10*28, 10*28))
plt.imshow(sample, cmap = plt.get_cmap('gray'), vmin=0, vmax=1)
plt.savefig('vae_samples_mnist/image_vae_mlp_')