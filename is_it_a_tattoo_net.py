from utilities import LoadData

update_file_names = LoadData.update_file_names

import numpy as np

import cPickle as pickle

import sys

import theano
import theano.tensor as T

from lasagne import layers
from lasagne.updates import nesterov_momentum,sgd
from lasagne.objectives import binary_crossentropy
from lasagne.nonlinearities import softmax,rectify
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
from lasagne.layers import Conv2DLayer as ConvLayer
#from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer 
#from lasagne.layers.dnn import Pool2DDNNLayer as PoolLayer
from lasagne.layers import MaxPool2DLayer as PoolLayer 
from lasagne.layers import LocalResponseNormalization2DLayer as NormLayer 
from lasagne.utils import floatX
from lasagne.init import GlorotUniform

from nolearn.lasagne import NeuralNet, BatchIterator

sys.setrecursionlimit(10000)

def float32(k):
    return np.cast['float32'](k)

def define_net(image_shape = (None,3,320,320),name='neural_net',verbose = 1):
    net = NeuralNet(
        layers =[
            ('input', InputLayer),
            ('conv1', ConvLayer),
            ('pool1', PoolLayer),
            ('conv2', ConvLayer),
            ('pool2', PoolLayer),
            ('conv3', ConvLayer),
            ('pool3', PoolLayer),
            ('conv4', ConvLayer),
            ('pool4', PoolLayer),
            ('dropout1', DropoutLayer),
            ('hidden5', DenseLayer),
            ('hidden6', DenseLayer),
            ('dropout2', DropoutLayer),
            ('hidden7', DenseLayer),
            ('dropout3', DropoutLayer),
            ('output', DenseLayer),
            ],
        input_shape = image_shape,
        conv1_num_filters    = 40,  conv1_filter_size = (15,15), 
        pool1_pool_size      = (2,2),
        conv2_num_filters    = 108, conv2_filter_size = (8,8), 
        pool2_pool_size      = (2,2),
        conv3_num_filters    = 260, conv3_filter_size = (3,3), 
        pool3_pool_size      = (2,2),
        conv4_num_filters    = 572, conv4_filter_size = (2,2), 
        pool4_pool_size      = (2,2),
        dropout1_p           = 0.2,
        hidden5_num_units    = 500,
        hidden6_num_units    = 500,
        dropout2_p           = 0.3,
        hidden7_num_units    = 250,
        hidden7_nonlinearity = rectify,
        dropout3_p           = 0.5,
        output_num_units     = 2,
        output_nonlinearity  = softmax,

        update               = sgd,

        update_learning_rate = theano.shared(float32(0.05)),
        #update_momentum     = theano.shared(float32(0.9)),

        regression           = False,

        batch_iterator_train = FlipBatchIterator(batch_size=75),

        on_epoch_finished    = [
            SaveBestAccScores(name),
            AdjustVariable('update_learning_rate',start=0.2,stop=0.001),
            #AdjustVariable('update_momentum',start=0.9,stop=0.999)
        ],

        max_epochs           = 2000,
        verbose              = verbose,
        )
    return net

class SaveBestAccScores(object):

    def __init__(self, name):
        self.best = 0
        self.name = name
        self.file_num = None #len(str())

    def __call__(self, nn, train_history):

        if self.file_num is None:
            digits = len(str(nn.max_epochs))
            file_num = '0:0{}d'.format(digits)
            self.file_num = '{' + file_num + '}'

        acc_score = train_history[-1]['valid_accuracy']
        if acc_score > self.best:
            self.best = acc_score
            file_string = self.file_num.format(train_history[-1]['epoch'])
            file_name = self.name + '_' + file_string +'.pkl'
            with open(file_name,'wb') as f:
                pickle.dump(nn,f)

class SaveEveryNIteration(object):
    def __init__(self, name ,N):
        self.N = N
        self.name = name
        self.count = 0

class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)

class FlipBatchIterator(BatchIterator):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def transform(self, Xb, yb):

        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        indices2 = np.random.choice(bs, bs / 2, replace=False)
        indices3 = np.random.choice(bs, bs / 2, replace=False)
        indices4 = np.random.choice(bs, bs / 2, replace=False)

        flipped = np.swapaxes(Xb[:,:,:,:],2,3)
        left = flipped[:,:,::-1,:]
        right = flipped[:,:,::-1,::-1]


        Xb[indices]  = Xb[indices, :, :, ::-1]
        Xb[indices2] = Xb[indices2, :, ::-1, :]
        Xb[indices3] = Xb[indices3, :, :, :]
        Xb[indices4] = Xb[indices4, :, :, :]

        return Xb,yb

if __name__ == '__main__':
    
    name = 'thirteenth_net'
    im_size = 160
    batch_size = 150

    net = define_net((None,3,im_size,im_size),name=name)

    # with open('twelth_net_final.pkl') as f:
    #     net_load = pickle.load(f)
    # net.load_params_from(net_load)
    
    net.batch_iterator_train = FlipBatchIterator( batch_size= batch_size )

    ld = LoadData(verbose=True)
    X,y =     ld.get_new_data(num_images = 750,
                              im_size = im_size,
                              crop_size = 160,
                              keep_crop_and_orig = True,
                              save_results = False,
                              save_sub_results = True,
                              file_name = 'processed__filtered_images_160.pkl.gz',
                              sub_names = ['filtered_urls_160.pkl.gz',
                                           'filtered_labels_160.pkl.gz',
                                           'data_raw_160.pkl.gz'],
                              overwrite_saves = True,
                              load_from_previous = False,
                              load_from_final = False,
                              previous_images = 'data_raw_160.pkl.gz',
                              previous_labels = 'filtered_labels.pkl.gz')



    net.fit(X,y)

    with open(name+'_final.pkl','wb') as f:
        pickle.dump(net,f)
    
