import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T
import timeit
import sys

from AdamTrainer import AdamTrainer
from datetime import datetime
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from theano.tensor.shared_randomstreams import RandomStreams

class AdversarialAdamTrainer(object):
    def __init__(self, rng, batchsize, 
                    enc_cost, dec_cost, disc_cost,
                    epochs=100, 
                    enc_alpha=0.00005, enc_beta1=0.75, enc_beta2=0.999, 
                    dec_alpha=0.00005, dec_beta1=0.75, dec_beta2=0.999, 
                    disc_alpha=0.00005, disc_beta1=0.75, disc_beta2=0.999, 

                    eps=1e-08, 
                    l1_weight=0.0, l2_weight=0.1, n_hidden_source = 100):

        self.enc_alpha = enc_alpha
        self.enc_beta1 = enc_beta1
        self.enc_beta2 = enc_beta2

        self.dec_alpha = dec_alpha
        self.dec_beta1 = dec_beta1
        self.dec_beta2 = dec_beta2
        
        self.disc_alpha = disc_alpha
        self.disc_beta1 = disc_beta1
        self.disc_beta2 = disc_beta2

        self.eps = eps
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.rng = rng
        self.theano_rng = RandomStreams(rng.randint(2 ** 30))
        self.epochs = epochs
        self.batchsize = batchsize
        self.n_hidden_source = n_hidden_source

        self.encoder = lambda network, x: network(x)        
        self.decoder = lambda network, x: network(x)
        self.discriminator = lambda network, x: network(x)

        self.encoder_cost = enc_cost
        self.decoder_cost = dec_cost
        self.discriminator_cost = disc_cost

    def randomize_uniform_data(self, n_input):
        return self.rng.uniform(size=(n_input, 800), 
                low=-np.sqrt(3, dtype=theano.config.floatX), 
                high=np.sqrt(3, dtype=theano.config.floatX)).astype(theano.config.floatX)


    def l1_regularization(self, network, target=0.0):
        return sum([T.mean(abs(p - target)) for p in network.params])

    def l2_regularization(self, network, target=0.0):
        return sum([T.mean((p - target)**2) for p in network.params])

    def get_cost_updates(self, enc_network, dec_network, disc_network, input, gen_rand_input):

        enc_result = self.encoder(enc_network, input)
        dec_result = self.decoder(dec_network, enc_result)
        disc_real_result = self.discriminator(disc_network, enc_result)
        disc_fake_result = self.discriminator(disc_network, gen_rand_input)

        # encoder update
        enc_cost = self.encoder_cost(disc_real_result, dec_result, input)

        enc_param_values = [p.value for p in self.enc_params]
        enc_gparams = T.grad(enc_cost, enc_param_values)

        enc_m0params = [self.enc_beta1 * m0p + (1-self.enc_beta1) *  gp     for m0p, gp in zip(self.enc_m0params, enc_gparams)]
        enc_m1params = [self.enc_beta2 * m1p + (1-self.enc_beta2) * (gp*gp) for m1p, gp in zip(self.enc_m1params, enc_gparams)]

        enc_params = [p - self.enc_alpha * 
                  ((m0p/(1-(self.enc_beta1**self.enc_t[0]))) /
            (T.sqrt(m1p/(1-(self.enc_beta2**self.enc_t[0]))) + self.eps))
            for p, m0p, m1p in zip(enc_param_values, enc_m0params, enc_m1params)]

        updates = ([( p,  pn) for  p,  pn in zip(enc_param_values, enc_params)] +
                   [(m0, m0n) for m0, m0n in zip(self.enc_m0params, enc_m0params)] +
                   [(m1, m1n) for m1, m1n in zip(self.enc_m1params, enc_m1params)] +
                   [(self.enc_t, self.enc_t+1)])

        # decoder update
        dec_cost = self.decoder_cost(dec_result, input)

        dec_param_values = [p.value for p in self.dec_params]
        dec_gparams = T.grad(dec_cost, dec_param_values)

        dec_m0params = [self.dec_beta1 * m0p + (1-self.dec_beta1) *  gp     for m0p, gp in zip(self.dec_m0params, dec_gparams)]
        dec_m1params = [self.dec_beta2 * m1p + (1-self.dec_beta2) * (gp*gp) for m1p, gp in zip(self.dec_m1params, dec_gparams)]

        dec_params = [p - self.dec_alpha * 
                  ((m0p/(1-(self.dec_beta1**self.dec_t[0]))) /
            (T.sqrt(m1p/(1-(self.dec_beta2**self.dec_t[0]))) + self.eps))
            for p, m0p, m1p in zip(dec_param_values, dec_m0params, dec_m1params)]

        updates += ([( p,  pn) for  p,  pn in zip(dec_param_values, dec_params)] +
                   [(m0, m0n) for m0, m0n in zip(self.dec_m0params, dec_m0params)] +
                   [(m1, m1n) for m1, m1n in zip(self.dec_m1params, dec_m1params)] +
                   [(self.dec_t, self.dec_t+1)])

        # discriminator update
        disc_cost = self.discriminator_cost(disc_fake_result, disc_real_result)

        disc_param_values = [p.value for p in self.disc_params]
        disc_gparams = T.grad(disc_cost, disc_param_values)

        disc_m0params = [self.disc_beta1 * m0p + (1-self.disc_beta1) *  gp     for m0p, gp in zip(self.disc_m0params, disc_gparams)]
        disc_m1params = [self.disc_beta2 * m1p + (1-self.disc_beta2) * (gp*gp) for m1p, gp in zip(self.disc_m1params, disc_gparams)]

        disc_params = [p - self.disc_alpha * 
                  ((m0p/(1-(self.disc_beta1**self.disc_t[0]))) /
            (T.sqrt(m1p/(1-(self.disc_beta2**self.disc_t[0]))) + self.eps))
            for p, m0p, m1p in zip(disc_param_values, disc_m0params, disc_m1params)]

        updates += ([( p,  pn) for  p,  pn in zip(disc_param_values, disc_params)] +
                   [(m0, m0n) for m0, m0n in zip(self.disc_m0params, disc_m0params)] +
                   [(m1, m1n) for m1, m1n in zip(self.disc_m1params, disc_m1params)] +
                   [(self.disc_t, self.disc_t+1)])

        return (enc_cost, dec_cost, disc_cost, updates)

    def train(self, enc_network, dec_network, disc_network, train_input, filename=None):

        """ Conventions: For training examples with labels, pass a one-hot vector, otherwise a numpy array with zero values.
        """
        
        # variables to store parameters
        self.enc_network = enc_network
        self.dec_network = dec_network
        self.disc_network = disc_network

        input = train_input.type()
        
        # Match batch index
        index = T.lscalar()
        rand_input = T.matrix()
        
        self.enc_params = enc_network.params
        enc_param_values = [p.value for p in self.enc_params]

        self.enc_m0params = [theano.shared(np.zeros(p.shape.eval(), dtype=theano.config.floatX), borrow=True) for p in enc_param_values]
        self.enc_m1params = [theano.shared(np.zeros(p.shape.eval(), dtype=theano.config.floatX), borrow=True) for p in enc_param_values]
        self.enc_t = theano.shared(np.array([1], dtype=theano.config.floatX))

        self.dec_params = dec_network.params
        dec_param_values = [p.value for p in self.dec_params]

        self.dec_m0params = [theano.shared(np.zeros(p.shape.eval(), dtype=theano.config.floatX), borrow=True) for p in dec_param_values]
        self.dec_m1params = [theano.shared(np.zeros(p.shape.eval(), dtype=theano.config.floatX), borrow=True) for p in dec_param_values]
        self.dec_t = theano.shared(np.array([1], dtype=theano.config.floatX))        

        self.disc_params = disc_network.params
        disc_param_values = [p.value for p in self.disc_params]

        self.disc_m0params = [theano.shared(np.zeros(p.shape.eval(), dtype=theano.config.floatX), borrow=True) for p in disc_param_values]
        self.disc_m1params = [theano.shared(np.zeros(p.shape.eval(), dtype=theano.config.floatX), borrow=True) for p in disc_param_values]
        self.disc_t = theano.shared(np.array([1], dtype=theano.config.floatX))

        enc_cost, dec_cost, disc_cost, updates = self.get_cost_updates(enc_network, dec_network, disc_network, input, rand_input)

        train_func = theano.function(inputs=[index, rand_input], 
                                     outputs=[enc_cost, dec_cost, disc_cost], 
                                     updates=updates, 
                                     givens={input:train_input[index*self.batchsize:(index+1)*self.batchsize],}, 
                                     allow_input_downcast=True)

        ###############
        # TRAIN MODEL #
        ###############
        print('... training')
        
        best_epoch = 0
        last_tr_mean = 0.

        start_time = timeit.default_timer()

        for epoch in range(self.epochs):
            
            train_batchinds = np.arange(train_input.shape.eval()[0] // self.batchsize)
            self.rng.shuffle(train_batchinds)

            sys.stdout.write('\n')

            tr_enc_costs  = []
            tr_dec_costs  = []
            tr_disc_costs = []
            for bii, bi in enumerate(train_batchinds):
                rand_data = self.randomize_uniform_data(self.batchsize)
                
                tr_enc_cost, tr_dec_cost, tr_disc_cost = train_func(bi, rand_data)

                sys.stdout.write('\r[Epoch %i]   enc cost: %.5f   dec cost: %.5f   disc cost: %.5f' % (epoch, tr_enc_cost, tr_dec_cost, tr_disc_cost))
                sys.stdout.flush()

                tr_enc_costs.append(tr_enc_cost)
                if np.isnan(tr_enc_costs[-1]):
                    print "NaN in encoder cost."
                    return

                tr_dec_costs.append(tr_dec_cost)
                if np.isnan(tr_dec_costs[-1]):
                    print "NaN in decoder cost."
                    return                    

                tr_disc_costs.append(tr_disc_cost)
                if np.isnan(tr_disc_costs[-1]):
                    print "NaN in discriminator cost."
                    return

            dec_network.save(filename)

            """
            self.enc_network.params = self.enc_params
            self.dec_network.params = self.dec_params

            gen_rand_input = theano.shared(self.randomize_uniform_data(100), name = 'z')

            generate_sample_images = theano.function([], self.decoder(self.dec_network, gen_rand_input))
            sampled_ae_gan = generate_sample_images()

            sampled_ae_gan = sampled_ae_gan.reshape((10,10,28,28)).transpose(1,2,0,3).reshape((10*28, 10*28))
            plt.imshow(sampled_ae_gan, cmap = plt.get_cmap('gray'), vmin=0, vmax=1)
            plt.savefig('samples_mnist_ae_gan/sample_AEGAN_' + str(epoch))
            """

        print
        print "Finished training..."