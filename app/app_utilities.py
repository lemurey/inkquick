import numpy as np

import matplotlib.pyplot as plt

import skimage.transform 

import urllib2

import requests

from StringIO import StringIO

from bs4 import BeautifulSoup

import requests


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

