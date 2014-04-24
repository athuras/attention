from sklearn.cluster import MiniBatchKMeans, KMeans
from scikits.talkbox.features import mfcc
import numpy as np

def mfcc_atomic(x, fs, nwin=256, nfft=512, nceps=10, drop=1):
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


def norm_ordered_centroids(x, km_kwargs={}):
    '''Return norm-ordered centroids'''
    km = MiniBatchKMeans(**km_kwargs)
    trained = km.fit(x)
    centroids = trained.cluster_centers_
    ind = np.argsort(np.linalg.norm(centroids, axis=1))
    return centroids[ind]

def pop_ordered_centroids(x, km_kwargs={}):
    '''Hack for now, hopefully it makes everything better...'''
    km = KMeans(**km_kwargs)
    trained = km.fit(x)
    centroids = trained.cluster_centers_
    popularity = np.bincount(trained.labels_)
    ind = np.argsort(popularity)
    return centroids[ind]

def filtered_mfcc_centroid(x, fs, filter_percentile=10,
        kmeans_kwargs={'n_clusters': 30, 'max_iter': 1000,
            'precompute_distances': True},
        mfcc_kwargs={'nwin':256, 'nfft':512, 'nceps':14}):
    '''Generates the centroids of the MFCC spectrogram of the signal x.
    1. Collect MFCC coefficients based on mfcc_kwargs,
    2. Calculate the deltas from MFCCs
    3. Filter out vectors smaller than *filter_percentile*,
    4. Normalise the MFCC coeffs (TODO: this),
    5. Calculate norm-ordered centroids using KMeans'''
    MFCCs = mfcc_atomic(x, fs, **mfcc_kwargs)
    features = stack_double_deltas(MFCCs)
    if filter_percentile != 0:
        features = low_energy_filter(features, filter_percentile)
    centroids = pop_ordered_centroids(features, km_kwargs=kmeans_kwargs)
    return centroids
