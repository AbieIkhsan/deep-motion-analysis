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

from tools.utils import load_cmu, load_cmu_small

rng = np.random.RandomState(23455)

BATCH_SIZE = 100

network = Network(
    
    Network(
        DropoutLayer(rng, 0.25),
        Conv1DLayer(rng, (64, 66, 25), (BATCH_SIZE, 66, 240)),
        Pool1DLayer(rng, (2,), (BATCH_SIZE, 64, 240)),
        ActivationLayer(rng, f='elu'),

        DropoutLayer(rng, 0.25),
        Conv1DLayer(rng, (128, 64, 25), (BATCH_SIZE, 64, 120)),
        Pool1DLayer(rng, (2,), (BATCH_SIZE, 128, 120)),
        ActivationLayer(rng, f='elu'),

        ReshapeLayer(rng, (BATCH_SIZE, 128*60)),
        DropoutLayer(rng, 0.25),    
        HiddenLayer(rng, (128*60, 1200)),
        ActivationLayer(rng, f='elu'),
    ),
    
    Network(
        VariationalLayer(rng, sample=False),
    ),
    
    Network(
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
)
shared = lambda d: theano.shared(d, borrow=True)
dataset, std, mean = load_cmu_small(rng)
E = shared(dataset[0][0])

network.load([[None, '../models/cmu/conv_varae/v_8/layer_0.npz', None, None, 
                None, '../models/cmu/conv_varae/v_8/layer_1.npz', None, None,
                None, None, '../models/cmu/conv_varae/v_8/layer_2.npz', None, ],
                [None,],
                [None, '../models/cmu/conv_varae/v_8/layer_3.npz', None, None,
                None, None, '../models/cmu/conv_varae/v_8/layer_4.npz', None,
                None, None, '../models/cmu/conv_varae/v_8/layer_5.npz', None],])

def cost(networks, X, Y):
    network_u, network_v, network_d = networks.layers
    
    vari_amount = 1.0
    repr_amount = 1.0
    
    H = network_u(X)
    mu, sg = H[:,0::2], H[:,1::2]
    
    # previous VAE
    #vari_cost = 0.5 * T.mean(mu**2) + 0.5 * T.mean((T.sqrt(T.exp(sg))-1)**2)
    #repr_cost = T.mean((network_d(network_v(H)) - Y)**2)
    
    #ya0st VAE
    vari_cost = 0.5 * T.sum(1 + 2 * sg - mu**2 - T.exp(2 * sg))
    repr_cost = T.sum((network_d(network_v(H)) - Y)**2)

    #return repr_amount * repr_cost + vari_amount * vari_cost
    return repr_amount * repr_cost - vari_amount * vari_cost

trainer = AdamTrainer(rng, batchsize=BATCH_SIZE, epochs=100, alpha=0.000001, cost=cost)
result = trainer.get_representation(network, E, 2)  * (std + 1e-10) + mean

new1 = result[2:3]
new2 = result[26:27]
new3 = result[0:1]

animation_plot([new1, new2, new3], interval=15.15)
