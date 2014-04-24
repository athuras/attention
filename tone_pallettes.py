# This is intended to be used in an interactive environment (a.k.a. ipython)
import numpy as np
import pandas as pd
import re
import subprocess
from collections import namedtuple
from itertools import ifilter

#  Constants
FNR = namedtuple('FileName', ['directory', 'lang', 'num', 'ext'])
Pattern = namedtuple('Pattern', ['lang', 'num', 'shape', 'data'])

def filename(record):
    return record.lang + str(record.num) + record.ext
def path(record):
    return record.directory + filename(record)


#  Replicated from feature_extraction.py (rather than imported).
def fn_parser(term):
    '''Parse the various information contained in the filename'''
    _SPLITTER = re.compile('([a-zA-Z]+)(\d+)(\..*)', re.IGNORECASE)
    lang, num, ext = _SPLITTER.findall(term)[0]
    return lang, int(num), ext

def filename_iter(directory, parser=fn_parser):
    '''Takes a directory, outputs a generator of parsed filenames'''
    fn_check = lambda x: not (x == 'error' or len(x) == 1)
    ls = subprocess.Popen(('ls', directory), stdout=subprocess.PIPE)
    q = subprocess.check_output(('grep', '.npy'), stdin=ls.stdout)
    ls.wait()
    filtered_filenames = ifilter(fn_check, (parser(l) for l in q.splitlines()))
    return (FNR(directory, *x) for x in filtered_filenames)

def load_pattern(record):
    '''FileNameRecord::record  -> namedtuple::Pattern'''
    data = np.load(path(record))
    shape = data.shape
    return Pattern(record.lang, record.num, shape, np.ravel(data))

def batch_extract_iter(directory):
    '''Return a generator of pattern tuples, which are inside of 'directory'.
    :directory -> must contain .npy or .npz (or pickled) numpy arrays!
    '''
    fni = filename_iter(directory, parser=fn_parser)
    return (load_pattern(r) for r in fni)

def dataset(directory):
    '''Return the patterns as a Pandas DataFrame,
    converts to list (hack, b/c pd.DataFrame.from_records doesn't
    play nice with generators'''
    data = list(batch_extract_iter(directory))
    columns = ['lang', 'num', 'shape', 'data']
    return pd.DataFrame.from_records(data, columns=columns)
