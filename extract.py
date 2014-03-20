# Feature extraction operating on mono audio

import numpy as np
from scikits.talkbox.features import mfcc
from scikits.learn.cluster import KMeans

def extract_mfcc_atomic(x, fs, nwin=256, nfft=512, nceps=13, drop=0):
    '''Extract mfcc from stream x.
    *drop: kick off the low-freq MFCCs, typically kill the DC coeff'''
    return mfcc(x, nwin, nfft, fs, nceps)[0][drop:]


def extract_mel_atomic(x, fs, nwin=256, nfft=512):
    '''Extract mel spectrum from stream x'''
    return mfcc(x, nwin, nfft, fs)[1]


def gen_extract(streams, f):
    for x, fs in streams:
        yield f(x, fs)


def gen_extract_mfcc(data_streams, *args):
    extract_fun = lambda x, fs: extract_mfcc_atomic(x, fs, *args)
    return gen_extract(data_streams, extract_fun)


def gen_extract_mel(data_streams, *args):
    extract_fun = lambda x, fs: extract_mel_atomic(x, fs, *args)
    return gen_extract(data_streams, extract_fun)


# Voice Filtering #############################################################


# K-Means Clustering ##########################################################

def cluster_centroids(x, k=32, max_iter=300, km_kwargs={}):
    '''Return norm-ordered centroids'''
    km = KMeans(k, init='k-means++', max_iter=300, **km_kwargs)
    trained = km.fit(x)
    centroids = trained.cluster_centers_
    ind = np.argsort(np.linalg.norm(centroids, axis=1))
    return centroids[ind]


def gen_cluster_samples(data_streams, k, km_kwargs={}):
    for x in data_streams:
        yield cluster_centroids(x, k, km_kwargs=km_kwargs)

def order_centroids(x):
    return np.linalg.norm(x)

# All around and back again ###################################################

def mfcc_centroids_from_raw(data_source, k=8, km_kwargs={},
                            mfcc_args=[]):
    '''Convenience Function to wrap everything else'''
    mfcc_gen = gen_extract_mfcc(data_source, *mfcc_args)
    cluster_gen = gen_cluster_samples(mfcc_gen, k, **km_kwargs)
    for x in cluster_gen:
        yield x

def mfcc_collection_as_matrix(source):
    '''Accumulate and concat,
    dimensions of resulting array (n, k, d)
    n = len(source),
    k = number of centroids
    d = dimension of centroid'''
    z = np.dstack(list(source))
    return z.swapaxes(0, 1).swapaxes(0, 2)
