# Script for batch feature extraction and archival.
import numpy as np
import multiprocessing
import argparse
from collections import Counter
from itertools import ifilter
from scipy.io import wavfile


_SPLITTER = re.compile('([a-zA-Z_]*(\d*)(\.\D*)', re.IGNORECARE)


def main(args):
    '''Extract features on the files in raw_audio_dir.
    Accumulate them, and write to disk as np.arrays of
    size (n, v), where n = num_of_samples, v = length of feature vector
    '''
    pass


# Get the accumulated language stats ##########################################
def fn_parser(term):
    '''Get the language'''
    lang, num, ext = _SPLITTER.findall(term)[0]
    return lang, num, ext


def process_file_list(directory, parser):
    '''Take no prisoners, only accept .wav files,
    returns a Histogram of language representation'''
    fn_check = lambda x: not (x == 'error' or len(x) == 1)
    q = subprocess.check_output(['ls' + directory + '| grep .wav'])
    parsed = (parser(l)[0] for l in q.splitlines())
    filtered = ifilter(fn_check, parsed)
    return Counter(filtered)


# Generate Tasks ##############################################################
def gen_task(term, max_work=10):
    '''Split histogram into manageable pieces:
    yields (lang, start_index, number) tuples that correspond to data files'''
    for lang, num in term:
        start = 1
        while num > 0:
            inc = min(num, max_work)
            yield lang, start, inc
            num -= inc
            start += inc


# Process Task ################################################################
def make_fn(lang, i, ext='.wav'):
    return lang + str(i) + ext


def task_to_file_iter(task):
    '''Divide the given task into its constituent jobs.
    Return iterator over filenames'''
    lang, start, num = task
    return (make_fn(lang, i) for i in xrange(start, start + num))


def read_to_array(fn, fs=22050):
    '''Read the file at fn, and downsample to 22kHz mono'''
    dfs, data = wavfile.read(fn)
    fac = fs / dfs
    downsampled = signal.resample(data, data.size * fac)
    return downsampled / 16.


def extract_feature(x, *args, **kwargs):
    pass
