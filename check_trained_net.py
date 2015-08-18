from utilities import (LoadData,
                       FlipBatchIterator,
                       SaveBestAccScores,
                       AdjustVariable,)

update_file_names = LoadData.update_file_names

import numpy as np

import cPickle as pickle

import sys

import time

if __name__ == '__main__':


    num_images = 750

    with open('best_net.pkl') as f:
        net = pickle.load(f)

    # load in files
    ld = LoadData(verbose=True)
    X,y =     ld.get_new_data(num_images = num_images,
                              im_size = 120,
                              keep_crop_and_orig = False,
                              load_from_previous = True,
                              previous_images = 'data_raw.pkl.gz',
                              previous_labels = 'filtered_labels.pkl.gz')
    # split off prediction files from full training/test data
    test_cut = num_images * 2 * 0.8
    Xp = X[test_cut:]
    yp = y[test_cut:]

    total_pics = float(len(X) - len(Xp))

    # start timer
    s_t = time.time()


    probs = net.predict_proba(Xp)
    cutoffs = np.linspace(0.27,0.29,10)
    for cutoff in cutoffs:
        score = []
        tp = 0.
        tn = 0.
        fp = 0.
        fn = 0.
        for i,prob in enumerate(probs):
            check = 0
            label = yp[i]
            if prob[1] > cutoff:
                check = 1
            if check == 1 and label ==1:
                tp += 1.
            elif check == 1 and label == 0:
                fp += 1.
            elif check == 0 and label == 0:
                tn += 1.
            else:
                fn += 1.

        accuracy = (tp+tn)/(tp+tn+fp+fn)
        print 'Accuracy for cutoff of {}: {}'.format(cutoff,accuracy)
        print 'False negative rate for cutoff of {}: {}'.format(cutoff,fn/total_pics)
        print '-'*80

    print 'took {}s to predict'.format(time.time() - s_t)