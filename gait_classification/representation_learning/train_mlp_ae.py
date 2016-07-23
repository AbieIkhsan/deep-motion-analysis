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
    HiddenLayer(rng, (784, 256)),
    ActivationLayer(rng, f='ReLU'),

    HiddenLayer(rng, (256, 64)),
    ActivationLayer(rng, f='ReLU'),

	HiddenLayer(rng, (64, 256)),
	ActivationLayer(rng, f='ReLU'),

	HiddenLayer(rng, (256, 784)),
	ActivationLayer(rng, f='sigmoid'),
)

trainer = AdamTrainer(rng=rng, batchsize=100, epochs=15, alpha=0.01, cost='mse')
trainer.train(network=network, train_input=train_set_x, train_output=train_set_x, 
    filename=None)

result = trainer.get_representation(network, train_set_x, 7)
print result.shape
sample = result[:100]

sample = sample.reshape((10,10,28,28)).transpose(1,2,0,3).reshape((10*28, 10*28))
plt.imshow(sample, cmap = plt.get_cmap('gray'), vmin=0, vmax=1)
plt.savefig('vae_samples_mnist/image_ae_mlp')