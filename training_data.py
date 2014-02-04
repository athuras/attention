# For English Data Only
# Properties can be changed by manipulating the globals: _LANG, _DATA_DIR

import numpy as np
import scipy.signal as signal
from scipy.io import wavfile

_LANG = 'english'
_DATA_DIR = './data'

def main():
    pass

## UTILITY ####################################################################

def normalize_data(x):
    return np.float64(x) / 16.

def resample(x, fs, rate=22050):
    '''Resample MONO signal to rate.
    returns (resmaple(signal), new_fs)'''
    fav = rate / fs
    return (signal.resample(x, x.size * fac), rate)


## TRAINING  ##################################################################

def gen_get_data(indices, lang, fs):
    '''Return an iterator over the parsed data streams'''
    for i in indices:
        x, dfs = get_data(i, lang)
        if dfs != fs:
            x, _ = resample(x, dfs, fs)
        yield x, fs

def get_data(k, language, norm_fun=normalize_data):
    '''Extract the signal from a wavfile, applying norm_fun to convert
    to float'''
    fs, data = wavfile.read(_DATA_DIR + '/' + language + str(int(k)) + '.mov.wav')
    data = norm_fun(data)
    return data, fs

if __name__ == '__main__':
    main()
