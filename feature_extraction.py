# Script for batch feature extraction and archival.
from collections import namedtuple
from features import filtered_mfcc_centroid
from itertools import ifilter
from multiprocessing import Pool
from scipy.io import wavfile
from scipy.signal import resample
import argparse
import numpy as np
import re
import subprocess
import time


#  take "english23.extension" and parse into "english", 23, ".wav.mpeg"
_SPLITTER = re.compile('([a-zA-Z_]*)(\d+)(\..*)', re.IGNORECASE)
FileName = namedtuple('FileName', ['lang', 'num', 'ext'])

def main(data_dir, output_dir, num_workers=2):
    # Get the files to process
    filenames = tasklist_iter('data/')
    pool = Pool(processes=num_workers)
    result = pool.map_async(composition, filenames)
    result.get()
    print "it is done."

#  Get the accumulated language stats #########################################
def fn_parser(term):
    '''Get the language'''
    lang, num, ext = _SPLITTER.findall(term)[0]
    return lang, num, ext

def tasklist_iter(directory, parser=fn_parser):
    '''Takes a directory, outputs a generator of parsed filenames'''
    fn_check = lambda x: not (x == 'error' or len(x) == 1)
    ls = subprocess.Popen(('ls', directory), stdout=subprocess.PIPE)
    q = subprocess.check_output(('grep', '.wav'), stdin=ls.stdout)
    ls.wait()
    filtered_filenames = ifilter(fn_check, (parser(l) for l in q.splitlines()))
    return (FileName(*x) for x in filtered_filenames)

#  Process Task ###############################################################
def read_to_array(fn, fs=22050):
    '''Read the file at fn, and downsample to 'fs' Hz mono'''
    dfs, data = wavfile.read(fn)
    fac = fs / dfs
    downsampled = resample(data, data.size * fac)
    return downsampled / 16.

def process_audio(x, fs=22050):
    '''Extract the Feature!'''
    return filtered_mfcc_centroid(x, fs, 10)

#  Disk-ops  ##################################################################
def fn_to_string(fn, with_ext=True):
    if with_ext:
        return fn.lang + str(fn.num) + fn.ext
    else:
        return fn.lang + str(fn.num)

def composition(fn_tuple, fs=22050):
    in_filename = 'data/' + fn_to_string(fn_tuple)
    out_filename = 'processed/' + fn_to_string(fn_tuple, with_ext=False)
    raw_audio = read_to_array(in_filename, fs=fs)
    tone_pallette = process_audio(raw_audio, fs=fs)
    np.save(out_filename, tone_pallette)
    print time.ctime, in_filename, ' -> ', out_filename

if __name__ == '__main__':
    desc = '''
    Given a directory of raw audio (*.wav) files, extract tone-pallettes.
    '''
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('data_directory', type=str,
            help="Directory containing the raw audio files in .wav format")
    parser.add_argument('output_directory', type=str,
            help="Directory to place the resulting .npy binary files")
    parser.add_argument('--num_workers', type=int, default=2,
            help="Limit the number of worker processes")
    args = parser.parse_args()
    main(args.data_directory, args.output_directory, limit=args.num_workers)
    main()
