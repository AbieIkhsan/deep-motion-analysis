import numpy as np
import sys
import theano
import theano.tensor as T

from ActivationLayer import ActivationLayer
from HiddenLayer import HiddenLayer

from theano.ifelse import ifelse
from theano.tensor.shared_randomstreams import RandomStreams

class LadderNetwork(object):
    """Implementation of the LadderNetwork as intoroduced in [1].
       Partially based on: https://github.com/rinuboney/ladder/
    
    References:
        [1] Rasmus, Antti, et al. 
        "Semi-Supervised Learning with Ladder Networks." 
        Advances in Neural Information Processing Systems. 2015.
        [2] Ioffe, Sergey, and Christian Szegedy. 
        "Batch normalization: Accelerating deep network training by reducing internal covariate shift." 
        arXiv preprint arXiv:1502.03167 (2015)."""

    def __init__(self, encoding_layers, decoding_layers, rng, sigma):

        if (0 == len(encoding_layers)):
            raise ValueError('Need to supply encoding layers')
        self.encoding_layers = encoding_layers 

        if (0 == len(decoding_layers)):
            raise ValueError('Need to supply decoding layers')
        self.decoding_layers = decoding_layers 

        if (0.0 >= sigma):
            raise ValueError('Received sigma <= 0.0')
        self.sigma = sigma

        # Used to add noise
        self.theano_rng = RandomStreams(rng.randint(2 ** 30))

        # Trainable parameters
        self.params = sum([e_layer.params for e_layer in self.encoding_layers], [])
        self.params += sum([d_layer.params for d_layer in self.decoding_layers], [])

        # Used to seperate between labeled from unlabeled examples
        self.labeled    = lambda x, y: x[T.nonzero(y)[0]]
        self.unlabeled  = lambda x, y: x[T.nonzero(1.-T.sum(y, axis=1))]
        self.split_data = lambda x, y: [self.labeled(x, y), self.unlabeled(x, y)]

        # Classification preidctions
        self.predictions = None

        # Used to calculate the unsupervised cost
        self.clean_z         = []
        self.reconstructions = []

        # As only hidden layers or conv layers encode units, we can simply
        # loop through all such layers and save the number of units
        n_units =  [l.output_units for l in self.encoding_layers if (hasattr(l, 'output_units'))]
        # Dimensionality of an input example 
        n_units.insert(0, self.encoding_layers[0].input_units)

        # Parameters needed to learn an optimal denoising function
        self.A = [theano.shared(value=np.ones((i, 10), dtype=theano.config.floatX), borrow=True) for i in n_units[::-1]]
        # Parameters of trainable batchnorm layers
        self.gamma = [theano.shared(value=np.ones((i, 10), dtype=theano.config.floatX), borrow=True) for i in n_units[1:]]
        self.beta = [theano.shared(value=np.zeros((i, 10), dtype=theano.config.floatX), borrow=True) for i in n_units[1:]]

        self.params += self.A
        self.params += self.gamma
        self.params += self.beta

    def __call__(self, input, output):
        """Calculates both a clean/noisy version of the encoder as well as the 
           corresponding denoising decoder. We pass both input and output to be able
           to split between labeled and unlabeled data points."""

        # Returns classification predictions and layerwise data
        noisy_y, noisy = self.fprop(input, output, sigma=self.sigma)
        # sigma = 0.0 -> no noise. Used to supply denoising targets
        self.predictions, clean = self.fprop(input, output, sigma=0.0)

        # Decoder part of the denoising autoencoder
        # TODO: noisy_y same dimension as noisy, clean?
        self.inv(noisy_y, noisy['unlabeled'], clean['unlabeled'])

        # Used to calculate the unsupervised cost
        self.clean_z = clean['unlabeled']['z_pre'].values()
        self.reconstructions = self.reconstructions[::-1]

        # predict with the clean version, calculate supervised cost with the noisy version
        return noisy_pass

    def add_noise(input, sigma=0.0):
        # Where sigma=0.0 is equivalent to a clean pass
        if (sigma > 0.0):
            gaussian_noise = sigma * self.theano_rng.normal(size=input.shape, dtype=theano.config.floatX)
            input += gaussian_noise

        return input

    def batchnorm(input, mean=None, std=None, eps=1e-10):
        """
            Performs batch-normalisation as proposed in [2]. Does not implement
            the trainable part including beta, gamma. This is done in self.beta_gamma.

            Parameters
            ----------
            input : Tensor 
                Minibatch to be normalised 
            mean : tensor
                std dev. of the incoming batch. Will be caclulated if not provided.
            std : tensor
                mean of the incoming batch. Will be caclulated if not provided.
                
        """
        # TODO: eval/training
        # Implements non-trainable batch-normalisation
        if (None != mean):
            mean = input.mean(0)
        if (None != std):
            std  = input.std(0)

        # Only batchnormalise for a batchsize > 1
        mean = ifelse(T.gt(input.shape[0], 1), mean, T.zeros(mean.shape, dtype=mean.dtype))
        std  = ifelse(T.gt(input.shape[0], 1), std, T.ones(std.shape, dtype=std.dtype))

        return (input - mean) / (std + eps)

    def beta_gamma(input, p_index):
        """
            p_index : int
                Used as an index for the trainable parameters beta, gamma.
                If p_index = -1, we implement non-traininable bn
        """
        if (p_index > -1):
            # Trainable bn
            bn = self.gamma[p_index] * (input + self.beta[p_index])

        return input

    def fprop(self, input, output, sigma=0.0):
        # Used to store pre-activations, activations, mean and std. dev of each layer
        d = {}
        d['labeled']   = {'z_pre': {}, 'mean': {}, 'std': {}}
        d['unlabeled'] = {'z_pre': {}, 'mean': {}, 'std': {}}

        input = self.add_noise(input, sigma)
        # 78
        d['labeled']['z_pre'][0], d['unlabeled']['z_pre'][0] = self.split_data(input)

        for id, layer in enumerate(self.encoding_layers):
            if (type(layer) is ActivationLayer):
                # labeled & unlabeld input
                l_input, u_input = self.split_data(input)

                u_mean = u_input.mean(0)
                u_std  = u_input.std(0)

                # Batch-normalise unlabeled and labeld examples seperatly 110/84-96
                l_input = self.batchnorm(l_input)
                u_input = self.batchnorm(u_input, u_mean, u_std)

                # TODO: save before or after noise?
                # Re-used during decoding (118-119)
                d['labeled']['z_pre'][id+1]   = l_input 
                d['unlabeled']['z_pre'][id+1] = u_input
                # statistics only for unlabeled examples 
                d['unlabeled']['mean'][id+1] = u_mean
                d['unlabeled']['std'][id+1]  = u_std

                # 114: TODO: Why only for softmax?
                input = self.add_noise(self.join(l_input, u_input), sigma)
                input = self.beta_gamma(input, id)

            input = layer(input)

        return input, d

    def compute_mu(self, input, A, offset=0):
        # Implements eq. (2) in [1] 
        o = offset
        return A[:,0+o] * T.nnet.sigmoid(input*A[:,1+o] + A[:,2+o]) + input*A[:,3+o] + A[:,4+o]

    def compute_v(self, input, A):
        return self.compute_mu(input, A, offset=5)
    
    def inv(self, noisy_y, u_noisy, u_clean):
        # Non batch-normalised estimates/resconstructions
        z_est = {}

        # decoding
        for d_index in range(0, len(self.decoding_layers)+1): 
            if (0 == d_index):
                u = noisy_y
            else:
                d_layer = self.decoding_layers[d_index-1]
                u = d_layer.inv(z_est[d_index-1])

            u = self.batchnorm(u)

            # Computations performed in the encoder
            noisy_z = u_noisy['z_pre'][-(d_index+1)]
            mean    = u_clean['mean'][-(d_index+1)]
            std     = u_clean['std'][-(d_index+1)]

            z_est[d_index] = self.skip_connect(u, noisy_z, d_index)

            # Used to calculate the denoising cost
            self.reconstructions.append(self.batchnorm(z_est[d_index], mean, std))

    def skip_connect(self, u, noisy_z, d_index):
        """ Implements the skip connections (g-function) in [1] eq. (2).
            These skip-connections release the pressure for higher layer to represent
            all information necessary for decoding.
        """

        mu = self.compute_mu(u, self.A[d_index])
        v  = self.compute_v(u, self.A[d_index])

        z_est = (noisy_z - mu) * v + mu

        return z_est
    
    def save(self, filename):
        if filename is None: return
        for fname, layer in zip(filename, self.encoding_layers + self.decoding_layers):
            layer.save(fname)
        
    def load(self, filename):
        if filename is None: return
        for fname, layer in zip(filename, self.encoding_layers + self.decoding_layers):
            layer.load(fname)
