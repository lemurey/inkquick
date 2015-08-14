from utilities import LoadData

update_file_names = LoadData().update_file_names

#download_images = LoadData().get_many_images

download_urls = LoadData().download_image_urls

import cPickle as pickle

import os

import matplotlib.pyplot as plt

from io import BytesIO

import threading

import urllib2

import requests

import math

from time import sleep,time

from PIL import Image

def rename_files(folder):
    s_t = time()
    for cur,subs,files in os.walk(folder):

        for i,f in enumerate(files):
            os.rename(f,'im_{}.jpg'.format(i))
    print 'renaming files took {} seconds'.format(time() - s_t)


def get_image_from_url(url,user,batch_num):
    '''
    Download am image from a provided url, and convert it to a numpy
    array that can be manipulated
    This has to be outside the class due to problem with multiprocessing
    not being able to pickle instance methods
    '''
    #ext = url.split('.')[-1]
    try:

        f = BytesIO(urllib2.urlopen(url).read())
        im = Image.open(f)
        im = im.resize((120,120))
        base = '/mnt/all_images/' 
        base += str(user) + '/' 
        base += 'image_'
        base += str(threading.current_thread().getName())
        file_name = base + '.jpg'

        d = os.path.dirname(file_name)
        if not os.path.exists(d):
            os.makedirs(d)
    except Exception as e:
        print e
        sleep(15)
        get_image_from_url(url,user,batch_num)
    im.save(file_name)

    #return 'image was saved'

def download_and_save_batch(list_urls,list_users,batch_size,batch_num):
    # pool = multiprocessing.Pool(processes = num_process)
    # cur_images = pool.map(get_image_from_url,list_urls)


    threads = [None] * batch_size

    sub_batch = batch_size/10.
    loops = int(math.ceil(batch_size/sub_batch))
    sub_batch = int(sub_batch)

    n = 0
    for k in range(loops):
        print 'minibatch started'
        threads = [None]*sub_batch
        for l in range(sub_batch):
            t = threading.Thread(target=get_image_from_url, args = (list_urls[n],list_users[n],batch_num))
            t.start()
            threads.append(t)
            n += 1
        for thread in threads:
            t.join()
        sleep(0.5)

def batch_download_images(batch_size,urls,users):
    total = len(urls)

    first_run = total/batch_size
    second_run = total - first_run*batch_size
    for i in xrange(first_run):
        s_t = time()
        start = i * batch_size
        stop = (i+1) * batch_size
        cur_urls = urls[start:stop]
        cur_users = users[start:stop]
        download_and_save_batch(cur_urls,cur_users,batch_size,i)
        print 'this loop took {}s {} loops to go'.format(time()-s_t,
                                                           first_run - i)
    stop = first_run * batch_size
    print stop
    if second_run > 0:
        s_t = time()
        cur_urls = urls[stop:][0]
        cur_users = urls[stop:][1]
        print len(cur_urls)
        download_and_save_batch(cur_urls,cur_users,second_run,first_run,)
        print 'final loop took {}s'.format(time() - s_t)

if __name__ == '__main__':
    
    urls =[]
    users = []

    csvs = []

    for f in os.listdir("./"):
        if f.endswith('.csv'):
            csvs.append(f)

    for j,csv in enumerate(csvs):
        s_t = time()

        with open(csv) as f:
            lines = f.readlines()

        print 'loading in csv{} file took {}s'.format(j,time()-s_t)

        for line in lines:
            if 'url_low' in line:
                continue
            vals = line.split(',')
            url = vals[0]
            user = vals[1].strip()
            urls.append(url)
            users.append(user)

    batch_download_images(2000,urls,users)
    rename_files('/mnt/all_images')

