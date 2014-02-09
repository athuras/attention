from scikits.learn.cluster import KMeans
from scikits.talkbox.features import mfcc
import numpy as np

def mfcc_atomic(x, fs, nwin=256, nfft=512, nceps=13, drop=0):
    return mfcc(x, nwin, nfft, fs, nceps)[0][drop:]


def norm_ordered_centroids(x, k=16, max_iter=300, km_kwargs={}):
    '''Return norm-ordered centroids'''
    km = KMeans(k, init='k-means++', max_iter=300, **km_kwargs)
    trained = km.fit(x)
    centroids = trained.cluster_centers_
    ind = np.argsort(np.linalg.norm(centroids, axis=1))
    return centroids[ind]

def mfcc_centroid(x, fs, kmeans_kwargs={'k':16, 'max_iter':300},
                 mfcc_kwargs={'nwin':256, 'nfft':512, 'nceps':13}):
