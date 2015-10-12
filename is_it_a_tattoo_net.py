from utilities import (LoadData,
                       float32,
                       SaveBestAccScores,
                       AdjustVariable,
                       FlipBatchIterator,)

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
'''
note: Conv2DDNNLayer only works on CudaDNN enabled systems
additionally Conv2DLayer can return different sizes from Conv2DDNNLayer
in some situations
I would recommend using Conv2DLayer 
'''
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer 
from lasagne.layers.dnn import Pool2DDNNLayer as PoolLayer

from nolearn.lasagne import NeuralNet, BatchIterator

sys.setrecursionlimit(10000) #<- neccesary for pickling neural net files

update_file_names = LoadData.update_file_names

def define_net(image_shape = (None,3,320,320),name='neural_net',verbose = 1):
    '''
    Define the layer structure of the neural_net used to determine
    whether or not an image is a tattoo
    '''

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





if __name__ == '__main__':
    
    #set name,image,and batch sizes
    name = 'tattoo_net'
    im_size = 120
    batch_size = 150

    # create net for training
    net = define_net((None,3,im_size,im_size),name =n ame)
    
    # update batch size for net
    net.batch_iterator_train = FlipBatchIterator( batch_size= batch_size )

    # load in data from heroku
    ld = LoadData(verbose=True)
    X,y =     ld.get_new_data(num_images = 750,
                              im_size = im_size,
                              crop_size = 160,
                              keep_crop_and_orig = True,
                              save_results = True,
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

    # fit model
    net.fit(X,y)

    # save final model
    with open(name+'_final.pkl','wb') as f:
        pickle.dump(net,f)
    
