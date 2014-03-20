from scikits.learn.cluster import KMeans
from scikits.talkbox.features import mfcc
import numpy as np

def mfcc_atomic(x, fs, nwin=512, nfft=1024, nceps=10, drop=0):
    return mfcc(x, nwin, nfft, fs, nceps)[0][:, drop:]

def stack_double_deltas(x):
    '''Stacks x on top of the various derivatives of x'''
    z = np.diff(x, axis=0)
    z2 = np.diff(z, axis=0)
    return np.hstack([x[:-2, ...], z[:-1, ...], z2])


def low_energy_filter(x, q, axis=1):
    '''Returns a view of x where each elements magnitude is greater than
    the q-th percentile.
    Within this context, I'm trying to filter out the 'quiet' segments of
    the recording, hopefully getting more signal coming from speech rather than
    rest'''
    norms = np.linalg.norm(x, axis=axis)
    p = np.percentile(norms, q)
    return x[norms > p]


def norm_ordered_centroids(x, k=20, max_iter=300, km_kwargs={}):
    '''Return norm-ordered centroids'''
    km = KMeans(k, init='k-means++', max_iter=300, **km_kwargs)
    trained = km.fit(x)
    centroids = trained.cluster_centers_
    ind = np.argsort(np.linalg.norm(centroids, axis=1))
    return centroids[ind]


def filtered_mfcc_centroid(x, fs, filter_percentile=10,
                  kmeans_kwargs={'k':30, 'max_iter':300},
                  mfcc_kwargs={'nwin':1024, 'nfft':2048, 'nceps':13}):
    '''Generates the centroids of the MFCC spectrogram of the signal x.
    1. Collect MFCC coefficients based on mfcc_kwargs,
    2. Calculate the deltas from MFCCs
    3. Filter out vectors smaller than *filter_percentile*,
    4. Normalise the MFCC coeffs (TODO: this),
    5. Calculate norm-ordered centroids using KMeans'''
    MFCSs = mfcc_atomic(x, fs, **mfcc_kwargs)
    features = stack_double_deltas(MFCCs)
    if filter_percentile != 0:
        features = low_energy_filter(features, filter_percentile)
    centroids = norm_ordered_centroids(features, **kmeans_kwargs)
    return centroids
