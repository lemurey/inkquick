from utilities import get_image_from_url,get_insta_url

from flask import Flask 
from flask import (request,
                   redirect,
                   url_for,
                   g,
                   session,
                   render_template
                  )

from forms import ImageForm

import pandas as pd

import cPickle as pickle

from run_prediction_model import run_model

from skimage.transform import resize

from numpy import swapaxes


app = Flask(__name__)
app.config.from_object('config')

with open('featurize_net.pkl','rb') as f:
    net = pickle.load(f) #<- load neural net for featurizing input images

# load in data frame of SF tattoo artists with featurized image vectors
df = pd.read_pickle('df_filtered_in_sf.pkl')

def get_results(url,df=df,nn=net):
    '''
    loads the image from the user submitted url, and then run the image through 
    the model
    '''
    image = get_image_from_url(url,resize=120,swap=True).reshape(-1,3,120,120)
    return run_model(df,nn,image)

@app.route('/',methods = ['GET','POST'])
def home():
    '''
    load homepage, redirect for results page if they have submitted a url
    '''
    form = ImageForm()
    if request.method == 'POST':
        url=form.image_url.data
        session['url'] = url
        return redirect(url_for('results'))
    return render_template('home.html',form=form)

@app.route('/results')
def results():
    '''
    get the submitted url from session, grab and featurize the image and 
    return the closest matches
    '''
    url = session['url']
    temp = url.split('/')
    if temp[2] == 'instagram.com':
        url = get_insta_url(url)
    print 'you submitted {} for image analysis'.format(url)
    results = get_results(url).index[:10].values

    #results = results.index[:5].values

    return render_template('results.html',results=results,image=url)


if __name__ == '__main__':
    
    app.run(host='0.0.0.0',port=5000,debug=True)