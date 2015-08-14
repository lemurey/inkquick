from utilities import LoadData

import numpy as np

import cPickle as pickle

import sys

import cPickle as pickle

import os,math

import matplotlib.pyplot as plt

import sqlite3 as sq

import multiprocessing

from time import time,sleep

from fourth_net import FlipBatchIterator,SaveBestAccScores,AdjustVariable

import sqlite3

def float32(k):
    return np.cast['float32'](k)

sys.setrecursionlimit(10000)

def get_files_for_users(user,files):
    out = []
    for f in files:
        if user == f[0]:
            out.append(f[1])

    return out

def open_and_reshape(f):
    im = plt.imread(f)
    im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)
    return (np.cast['float32'](im / 255),f)

def get_batches(files,num_batches,pool,batch_size):
    start = 0
    stop = 0
    out = []
    for batch in range(num_batches):
        stop += int(batch_size)
        to_get = files[start:stop]
        imgs = pool.map(open_and_reshape,to_get)
        start += int(batch_size)
        out.extend(imgs)

    return out


if __name__ == '__main__':
    '''
    predict_proba[0] is probability of 0
    '''

    name = 'testing_net'
    im_size = 120
    # batch_size = 150

    with open('twelth_net_0143.pkl') as f:
        net = pickle.load(f)


    s_t = time()

    yes_cutoff = 0.65
    batch_size=5000.


    image_files = [None] * 502759



    conn = sqlite3.connect('predictions_2.db')
    c = conn.cursor()
    
    c.execute('''
              CREATE TABLE predictions
              (id integer, image_num text, probability float)
              ''')

    conn.commit()

    start_folder = '/home/ubuntu/orig_images'
    i = 0
    for cur,subs,files in os.walk(start_folder):
        if cur == start_folder:
            users = subs
        cur_user = cur.split('/')[-1]


        for f in files:
            image_files[i] = (cur_user,f)
            i += 1

    pool = multiprocessing.Pool(processes = 1000)

    print 'setup took {} seconds'.format(time() - s_t)

    total_tattoos = 0.
    for i,user in enumerate(users):
        print '-'*80
        print 'starting new user: {}'.format(user)
        s_t = time()
        paths = []
        db_updates = []
        percent_updates = []
        num_tattoos = 0.
        files = get_files_for_users(user,image_files)
        to_get = len(files)

        num_batches = int(math.ceil(to_get/batch_size))
    

        for f in files:
            paths.append(os.path.join(start_folder,user,f))

        images = get_batches(paths,num_batches,pool,batch_size)

        for image,f in images:
            image = np.array(image).reshape(1,3,120,120)
            prob = float(net.predict_proba(image)[0][1])
            image_id = f.split('/')[-1]
            db_updates.append((user,image_id,prob))


        
        # print images.shape

        # for image in images
        # probs = net.predict_proba(images)

        # for p in probs:
        #     prob = p[1]
        #     pred = 0
        #     if prob > yes_cutoff:
        #         num_tattoos += 1.
        #         pred = 1
        #     db_updates.append((user,pred))


        print 'predicting took {:.3f} seconds'.format(time() - s_t)
        print 'number of users remaining: {}'.format(496 - i)
        c.executemany('INSERT INTO predictions VALUES (?,?,?)',db_updates)

    conn.commit()

    print 'all done'
