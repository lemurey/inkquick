import os

import numpy as np

import matplotlib.pyplot as plt

import skimage.transform 

import urllib2

import requests

import time

import multiprocessing

import cPickle as pickle

import gzip

from StringIO import StringIO

import Queue

import threading

from bs4 import BeautifulSoup

import requests

from nolearn.lasagne import BatchIterator

def get_insta_url(url):
    '''
    function to convert instagram short urls to links to the raw jpg image
    '''

    z = requests.get(url)
    soup = BeautifulSoup(z.content,'html5lib')
    try:
        string = str(soup.findAll(attrs={'property':'og:image'})[0])
    except:
        string = '<meta content="property was not there"'
    out = string.split('<meta content="')[1].split('"')[0]
    temp = out.split('/')
    #remove taken by tag if it is there
    if temp[-1][0] == '?':
        del temp[-1]
    #convert from to 320X320 image size url (unless it already is)
    if '/s320x320/' not in temp:
        out = '/'.join(temp[:-1]) + '/s320x320/' + temp[-1]
    else:
        out = '/'.join(temp)
    return out


def float32(k):
    '''
    helper function to recast numpy arrays as type float32
    float32 is needed for input to theano based neural nets
    '''
    return np.cast['float32'](k)

def get_image_from_url(url,resize=None,swap=False):
    '''
    Download am image from a provided url, and convert it to a numpy
    array that can be manipulated
    resize and swap values can be passed to do inplace resizing and swapping
    of axes for input to neural nets
    This has to be outside the class due to problem with multiprocessing
    not being able to pickle instance methods
    '''
    ext = url.split('.')[-1]
    f_url = StringIO(urllib2.urlopen(url).read())
    im = plt.imread(f_url,ext)
    if resize:
        im = skimage.transform.resize(im, (resize,resize),)
    if swap:
        im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)
    return float32(im)




class LoadData(object):


    def __init__(self,verbose = True):
        # set these to be urls to image database
        self.image_url = 'url_for_images_db'
        self.all_images = 'url_for_other_image_db'
        self.verbose = verbose

    def resize_image(self,im,size):
        ''' 
        resize the image to be square and shape size,size
        this has to be done before reshapeing the image
        '''
        im = skimage.transform.resize(im, (size,size),)
        return im 

    def reshape_image(self,im):
        ''' 
        change the image from being shape (height, width, channels)
        to being shape (channels, height, width)
        '''
        im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)
        return im

    def center_crop_image(self,im_in,size):
        ''' 
        crop out the middle of the image of shape size by size
        '''
        im = im_in.copy()
        h,w,_ = im.shape
        if size >= h or size >= w:
            print 'Cannot have a crop bigger than original image'
            return im
        start_h = (h - size)//2
        end_h = (h + size)//2
        start_w = (w - size)//2
        end_w = (w + size)//2
        im = im[start_h:end_h,start_w:end_w,:]
        return im

    def download_image_urls(self,num_images,all_images = False):
        ''' 
        downloads an equal number of images coded as yes tattoo and no
        tattoo from the heroku app that hosts my image coding routine
        note that num_images is the number of yes images returned, so you
        actually get 2 * num_images results
        '''

        if all_images:
            in_url = self.all_images
            results = [None]* 504677
        else:
            in_url = self.image_url + str(num_images)
            results = ([None]*2*num_images,[None]*2*num_images)

        raw = requests.get(in_url)
        
        if len(results) != 2:
            url_out = self.parse_download_all(results,raw)
            return (url_out , None)

        i = 0
        for line in raw.content.split('\n'):
            data = line.split(',')
            if len(data) == 1:
                continue
            url = data[0].split('<p>')[1]
            code = data[1].split('</p>')[0]
            results[0][i] = url
            results[1][i] = code
            i += 1
        return results

    def parse_download_all(self,results,raw):
        '''
        function to parse data from the entire photo database
        '''
        i = 0
        for line in raw.content.split('\n'):
            if '<p>' in line:
                url = line.split('<p>')[1].split('</p>')[0]
                results[i] = url
        return results

    def save_to_file(self,inputs,file_name,overwrite=False):
        '''
        save the input to a gzipped pickled file, if overwrite is False, which
        is the default behaviour, then the file_name given has a number
        appended to it before saving, leaving both the original file and the
        new one, otherwise no checking for existence of file is done
        '''
        if not overwrite:
            check = file_name.split('.')
            if len(check) > 2:
                ext = '.'.join(check[-2:])
                base = '.'.join(check[0:-2])
            else:
                ext = '.'.join(check[-1:])
                base = '.'.join(check[0:-1])
            file_name = self.update_file_names(base,ext)
        with gzip.open(file_name,'wb') as f:
            pickle.dump(inputs,f)

    def load_from_file(self,file_name,gzipped = True):
        '''
        load the data from the given zipped pickled file
        '''
        if not os.path.isfile(file_name):
            print 'File not found, please check file name provided'
            return None
        if gzipped:
            with gzip.open(file_name,'rb') as f:
                out = pickle.load(f)
        else:
            with open(file_name,'rb') as f:
                out = pickle.load(f)

        return out

    def update_file_names(self,file_name,ext):
        '''
        updates file names with sequential numbers if they already exist in
        the current directory
        '''
        searching = True
        outstring = file_name
        addon = 0
        while searching:
            if os.path.isfile(outstring + ext):
                addon += 1
                outstring = file_name + str(addon)
            else:
                searching = False
        return outstring + '.' + ext

    def get_many_images(self,list_urls,num_process = 1,display = True):
        '''
        downloads every image in a list of urls, can use multiproccessing to
        download them in parallel if a value is specified for num_process
        display option reports how long the downloading took
        '''

        s_t = time.time()

        pool = multiprocessing.Pool(processes = num_process)
        out = pool.map(get_image_from_url, list_urls)

        if display:
            print 'Downlading images took {:.3f}s'.format(time.time() - s_t)

        return out

    def load_from_directory(self,start_dir):
        '''
        load data from a directory where either all the images you want are
        in that directory or in in subdirectories,returns a list of tuples with 
        each directory and the images in that directory
        '''

        out = []
        for cur,subs,cur_files in os.walk(start_dir):
            images = []
            if cur == start_dir:
                continue
            for f in cur_files:
                get_path = os.path.join(cur,f)
                im = plt.imread(get_path)
                im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)
                im = np.cast['float32'](im / 255)
                images.append(im)
            out.append((cur,images))

        
        return out

    def get_new_data(self,
                     num_images = 700,
                     im_size = None,
                     crop_size = None,
                     keep_crop_and_orig = True,
                     save_results = False,
                     save_sub_results = False,
                     num_process = None,
                     file_name = 'processed_images.pkl.gz',
                     sub_names = ['X.pkl.gz','y.pkl.gz','raw_images.pkl.gz'],
                     overwrite_saves = False,
                     load_from_previous = False,
                     load_from_final = False,
                     previous_images = '',
                     previous_labels = '',
                     all_images = False):
        '''
        gets data from scratch. If save_results is set to true  then
        the final processed data is saved. If save_sub_results is set to true
        then it saves the urls,codes, and unproccesed images as
        well. It is possible to only save part of the sub_results by setting
        the sub_names list elements to None for the sub_results you do not want
        saved. If im_size is specified the images are set to square size of
        size im_size. If crop_size is set, the images are center cropped to the
        specified size, if im_size is also set this is done before resizing.
        If keep_crop_and_orig is set to true (default) then the original image
        and the center cropped images are both saved.
        By default a number of processes equal to num_images is used for 
        downloading, this ccan be modified by passing a number to num_process
        If overwrite_saves is set to true, then any save operations will 
        overwrite existing files
        '''

        s_t = time.time()

        if load_from_previous:
            with gzip.open(previous_images) as f:
                images = pickle.load(f)
            with gzip.open(previous_labels) as f:
                labels = pickle.load(f)
            if self.verbose:
                print 'loading images and labels took {:.3f}s'.format(time.time() - s_t)

        else:

            urls, labels = self.download_image_urls(num_images,all_images)

            if not num_process:
                num_process = num_images

            images = self.get_many_images(urls,num_process,self.verbose)

            if self.verbose:
                print 'downloading images and labels took {:.3f}s'.format(time.time() - s_t)

            s_t = time.time()
            if save_sub_results:
                if sub_names[0]:
                    self.save_to_file(urls,sub_names[0],overwrite_saves)
                if sub_names[1]:
                    self.save_to_file(labels,sub_names[1],overwrite_saves)
                if sub_names[2]:
                    self.save_to_file(images,sub_names[2],overwrite_saves)
                if self.verbose:
                    print 'saving sub_data took {:.3f}s'.format(time.time() - s_t)



        s_t = time.time()

        if crop_size:
            temp_images = [None] * len(images)
            for i,image in enumerate(images):
                    temp_images[i] = self.center_crop_image(image,crop_size)
            # keeps the cropped and original image, allowing for double training
            # on the center of images (kind of)
            if keep_crop_and_orig:
                images.extend(temp_images)
                labels.extend(labels)
            else:
                images = temp_images
            if self.verbose:
                print 'images cropped'

        if im_size:
            for i,image in enumerate(images):
                images[i] = self.resize_image(image,im_size)
            if self.verbose:
                print 'images resized'

        if not load_from_final:
            for i,image in enumerate(images):
                images[i] = self.reshape_image(image)
            if self.verbose:
                print 'images reshaped'

        if self.verbose:
            print 'modifying images took {:.3f}s'.format(time.time() - s_t)

        if not load_from_final:
            images = np.array(images).astype(np.float32)
            labels = np.array(labels)
            labels = (labels == 'yes') * 1
            labels = np.array(labels).astype(np.int32)
        else:
            labels = images[1]
            images = images[0]

        s_t = time.time()

        if save_results:
            self.save_to_file((images,labels),file_name,overwrite_saves)
            if self.verbose:
                print 'saving data took {:.3f}s'.format(time.time() - s_t)

        return images,labels

def update_nums(in_str):
    '''
    much more detailed version of file name updating
    '''
    parts = in_str.split('.')

    end = parts[-1]

    beginning ='.'.join(parts[:-1])

    if len(parts) == 1:
        beginning = end

    num_digits = 0
    end_num = ''

    for char in beginning[::-1]:
        if char.isdigit():
            num_digits += 1
            end_num += char

    prep_val = '{' + '0:0{}d'.format(num_digits) + '}'

    if len(end_num) == 0:
        new_num = 1
        num_digits = -len(beginning)
        new_identifier = '_'
    else:
        new_num  = int(end_num[::-1]) + 1
        new_identifier = ''

    new_identifier += prep_val.format(new_num)
    

    if len(parts) == 1:
        out = beginning + new_identifier
    else:
        out = beginning[:-num_digits] + new_identifier + '.' + end

    return out

class SaveBestAccScores(object):

    def __init__(self, name):
        self.best = 0
        self.name = name
        self.file_num = None #len(str())

    def __call__(self, nn, train_history):
        '''
        check if current epochs accuracy score is better than previous 
        epochs, if it is, save the nn for the current epoch
        '''
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

class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        '''
        code mostly taken from DNouri: https://github.com/dnouri/nolearn
        used to update learning paramaters during training 
        '''
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)

class FlipBatchIterator(BatchIterator):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def transform(self, Xb, yb):
        '''
        code adapted from DNouri:https://github.com/dnouri/nolearn
        used to flip images during training to expand dataset
        does x and y axes flips along with 90 and 270 degree rotations
        '''
        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        indices2 = np.random.choice(bs, bs / 2, replace=False)
        indices3 = np.random.choice(bs, bs / 2, replace=False)
        indices4 = np.random.choice(bs, bs / 2, replace=False)

        flipped = np.swapaxes(Xb[:,:,:,:],2,3)
        left = flipped[:,:,::-1,:]
        right = flipped[:,:,::-1,::-1]

        # comment any of the below lines to remove flipping of those 
        # axes/directions
        Xb[indices]  = Xb[indices, :, :, ::-1]
        Xb[indices2] = Xb[indices2, :, ::-1, :]
        Xb[indices3] = left[indices3, :, :, :]
        Xb[indices4] = right[indices4, :, :, :]

        return Xb,yb

if __name__ == '__main__':
    


    ld = LoadData()
    all_urls = ld.download_image_urls(100,True)
    print len(all_urls)
    print all_urls[-1]






