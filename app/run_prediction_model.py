from scipy.spatial.distance import cosine

from functools import partial

from time import time

def run_model(df,nn,im):
    '''
    featurize an imported image and then return the distance to all images in
    my collection, grouped by which account posted the images
    '''

    s_t = time()
    feature_vec = nn.predict_proba(im)
    print 'featurizing took {} seconds'.format(time() - s_t)

    s_t = time()
    pd_cosine = partial(cosine,feature_vec)

    df['distance'] = df['features'].apply(pd_cosine)

    out = df.groupby('name').mean().sort('distance')

    return out


