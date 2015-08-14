from collections import OrderedDict

import math
import operator

def get_size(layer_type, sizes,prev_layer_size = None):
    #print layer_type
    if layer_type == 'drop':
        #print 'it is a dropout layer'
        return prev_layer_size
    elif layer_type == 'conv':
        first  = sizes[0]
        second = prev_layer_size[1] - sizes[1] + 1
        third  = second
        #print 'it is a covolutional layer'
        return (first,second,third)
    elif layer_type == 'pool':
        first  = sizes[0]
        second = int(math.ceil(prev_layer_size[1]/2.))
        third  = second
     #   print 'it is a pooling layer'
        return (first,second,third)
    else:
     #   print 'it is an input or hidden layer'
        return sizes


def predict_layer_structure(input_size   = (3,120,120),
                            first_filter = 40,
                            multiples    = [1,5,23.9,105],
                            filters      = [15,8,3,2],
                            pools        = [2,2,2,2]):
    

    layer_structure = OrderedDict()
    layer_structure['input']    = input_size
    for i in range(len(multiples)):
        layer_name = 'conv{}'.format(i+1)
        value = int(first_filter*multiples[i])
        layer_structure[layer_name] = [ value , filters[i]]
        layer_name = 'pool{}'.format(i+1)
        layer_structure[layer_name] = [ value , pools[i]]

    # layer_structure['conv1']    = [first_filter           , 15]
    # layer_structure['pool1']    = [first_filter           ,  2]
    # layer_structure['conv2']    = [int(first_filter*5)    ,  8]
    # layer_structure['pool2']    = [int(first_filter*5)    ,  2]
    # layer_structure['conv3']    = [int(first_filter*23.9) ,  3]
    # layer_structure['pool3']    = [int(first_filter*23.9) ,  2]
    # layer_structure['conv4']    = [int(first_filter*105)  ,  2]
    # layer_structure['pool4']    = [int(first_filter*105)  ,  2]
    layer_structure['dropout2'] = []
    layer_structure['hidden5']  = [500]
    layer_structure['hidden6']  = [500]
    layer_structure['dropout3'] = [500]
    layer_structure['hidden7']  = [250]
    layer_structure['dropout4'] = [250]
    layer_structure['output']   = [2]

    breakpoints = (3,8,9,9)
    layer_sizes = []
    print '  #  name\t   size\t\t  total num '
    print '  '.join(['-'*i for i in breakpoints])
    i = 0

    for layer,sizes in layer_structure.iteritems():
        if i == 0:
            size = get_size(layer[:4],sizes)
        else:
            size = get_size(layer[:4],sizes,size)

        if len(size) > 1:
            layer_num = 1
            cur_size = 'x'.join(str(j) for j in size)
            for entry in size:
                layer_num *= entry
                # layer_sizes.append(entry)
        else:
            cur_size = size[0]
            layer_num = size[0]
            # layer_sizes.append(size[0])

        
        

        first  = '{0: 3d}'.format(i)
        dis1 = breakpoints[0]-len(first)
        second =  ' '*(dis1 + 2) + layer
        dis2 = breakpoints[1] - len(layer)
        third  =  ' '*(dis2 + 2) + str(cur_size)
        dis3 = breakpoints[2] - len(str(cur_size))
        fourth =  ' '*(dis3 + 2) + str(layer_num)

        print first + second + third + fourth
        i += 1

    #print layer_sizes
    #print 'total size: {}'.format(sum(layer_sizes))
    #print len(str(reduce(operator.mul,layer_sizes,1)))

    # for layer,sizes in layer_structure.iteritems():
    #     print sizes
    #     if sizes != 0:
    #         layer_size = sizes
    #     #else:




    #     #layer_sizes.append(layer_size)
    # print 'total size: ', sum(layer_sizes)

if __name__ == '__main__':
    predict_layer_structure(input_size   = (3,120,120),
                            first_filter = 40,
                            multiples    = [1,2.7,6.5,14.3],
                            filters      = [15,8,3,2],
                            pools        = [2,2,2,2])
