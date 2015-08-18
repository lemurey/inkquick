from utilities import float32,get_image_from_url

import cPickle as pickle

import theano



from lasagne import layers
from lasagne.updates import sgd
from lasagne.nonlinearities import rectify
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer 
from lasagne.layers.dnn import Pool2DDNNLayer as PoolLayer

from nolearn.lasagne import NeuralNet

def define_net(image_shape = (None,3,120,120),name='neural_net',verbose = 1):
    '''
    make neural net that has no dropout or classification layers, but is 
    otherwise identical to the is it a tattoo or nor trained network
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
            ('hidden5', DenseLayer),
            ('hidden6', DenseLayer),
            ('hidden7', DenseLayer),
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
        hidden5_num_units    = 500,
        hidden6_num_units    = 500,
        hidden7_num_units    = 250,
        hidden7_nonlinearity = rectify,

        update               = sgd,

        update_learning_rate = theano.shared(float32(0.05)),

        regression           = False,


        max_epochs           = 2000,
        verbose              = verbose,
        )
    return net


if __name__ == '__main__':

    # define net with no classification layer
    net = define_net()

    # load in best parameters
    net.load_params_from('featurize_params.pkl')

    # save featurization network
    with open('featurize_net.pkl','wb') as f:
        pickle.dump(net,f)
