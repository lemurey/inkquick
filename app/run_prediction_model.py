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

    #best = df.groupby('name')['distance'].min()
    out = df.groupby('name').mean().sort('distance')



    # vals = df.groupby('name').apply(lambda x: x.iloc[x.distance.idxmin()])
    # print vals

    # print 'calculating distance took {} seconds'.format(time() - s_t)
    best = []
    # top_ids = out['id'][:5].values

    # shortest_distance = best[best['id'].isin(top_ids)].distance

    # print shortest_distance

    # f_names = []

    return (out,best)


