import numpy as np
import theano
import theano.tensor as T

import sys
sys.path.append('../representation_learning/')

#from nn.ActivationLayer import ActivationLayer
#from nn.AdamTrainer import AdamTrainer
#from nn.BatchNormLayer import BatchNormLayer
#from nn.Conv1DLayer import Conv1DLayer
#from nn.HiddenLayer import HiddenLayer
#from nn.Network import Network
#from nn.Pool1DLayer import Pool1DLayer
#from nn.ReshapeLayer import ReshapeLayer

rng = np.random.RandomState(23455)

data = np.load('../data/data_styletransfer.npz')

#(Examples, Time frames, joints)
clips = data['clips']

clips = np.swapaxes(clips, 1, 2)
X = clips[:,:-4]

#(Motion, Styles)
classes = data['classes']

# get mean and std
preprocessed = np.load('../data/styletransfer_preprocessed.npz')

Xmean = preprocessed['Xmean']
Xmean = Xmean.reshape(1,len(Xmean),1)
Xstd  = preprocessed['Xstd']
Xstd = Xstd.reshape(1,len(Xstd),1)

Xstd[np.where(Xstd == 0)] = 1

X = (X - Xmean) / Xstd

# Motion labels in one-hot vector format
Y = np.load('../data/styletransfer_motions_one_hot.npz')['one_hot_vectors']

# Randomise data
shuffled = zip(X,Y)
np.random.shuffle(shuffled)

cv_split   = int(X.shape[0] * 0.6)
test_split = int(X.shape[0] * 0.8)

X, Y = map(np.array, zip(*shuffled))
X_train = theano.shared(np.array(X)[:cv_split], borrow=True)
Y_train = theano.shared(np.array(Y)[:cv_split], borrow=True)

X_valid = theano.shared(np.array(X)[cv_split:test_split], borrow=True)
Y_valid = theano.shared(np.array(Y)[cv_split:test_split], borrow=True)

X_test = theano.shared(np.array(X)[test_split:], borrow=True)
Y_test = theano.shared(np.array(Y)[test_split:], borrow=True)

network = Network(
    Conv1DLayer(rng, (64, 66, 25), (1, 66, 240)),
    Pool1DLayer(rng, (2,), (1, 64, 240)),
    ActivationLayer(rng, f='ReLU'),

    Conv1DLayer(rng, (128, 64, 25), (1, 64, 120)),
    Pool1DLayer(rng, (2,), (1, 128, 120)),
    ActivationLayer(rng, f='ReLU'),
    
    Conv1DLayer(rng, (256, 128, 25), (1, 128, 60)),
    Pool1DLayer(rng, (2,), (1, 256, 60)),
    ActivationLayer(rng, f='ReLU'),

    # Final pooling gives 1, 256, 30 

    # 256*60 = 7680
    ReshapeLayer(rng, (7680, )),
    HiddenLayer(rng, (np.prod([1, 256, 30]), 8)),
    ActivationLayer(rng, f='softmax'),
)

# Load the pre-trained conv-layers
network.load(['../models/layer_0.npz', None, None,
              '../models/layer_1.npz', None, None,
              '../models/layer_2.npz', None, None, 
              None, None, None])

trainer = AdamTrainer(rng=rng, batchsize=1, epochs=1, alpha=0.1, cost='cross_entropy')

# Fine-tuning for classification
trainer.train(network=network, train_input=X_train, train_output=Y_train, 
                               valid_input=X_valid, valid_output=Y_valid,
                               test_input=X_test, test_output=Y_test,
                               # Don't save the params
                               filename=[None] * len(network.layers))

# Don't save anyting for now
#filename=[None,
#'layer_0_finetuned.npz', None, None,
#'layer_1_finetuned.npz', None, None,
#'layer_2_finetuned.npz', None, None,
#None, 'layer_3_finetuned.npz', None,])
