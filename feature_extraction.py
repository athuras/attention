# Script for batch feature extraction and archival.
from collections import namedtuple
from features import filtered_mfcc_centroid
from functools import partial
from itertools import ifilter, imap
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

def main(data_dir, output_dir, archive_dir, num_workers=2):
    # Get the files to process
    filenames = tasklist_iter(data_dir)
    pool = Pool(processes=num_workers)
    do_work = partial(composition,
                    fs=22050,
                    input_dir=data_dir,
                    output_dir=output_dir,
                    archive_dir=archive_dir)
    result = pool.map_async(do_work, filenames)
    status = result.get()
    with open('feature_extraction.log', 'w') as f:
        f.writelines(imap(str, status))
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
def pad_to_nearest_2pow(x):
    desired_size = 2**int(np.ceil(np.log2(x.size)))
    padding = np.zeros(desired_size - x.size)
    return np.concatenate((x, padding), axis=0)

def read_to_array(fn, fs=22050):
    '''Read the file at fn, and downsample to 'fs' Hz mono'''
    dfs, data = wavfile.read(fn)
    # WARNING: this is a hack to make sure the downsampling is fast
    # this WILL introduce high-frequency components, however they should
    # vanish after downsampling.
    data = pad_to_nearest_2pow(data)
    new_size = int(data.size * fs / float(dfs))
    downsampled = resample(data, new_size)
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

def move_when_done(in_filename, archive_dir='data_archive/'):
    ''' Runs: $(mv in_filename archive_dir)'''
    subprocess.check_output(('mv', in_filename, archive_dir))

def composition(fn_tuple, fs=22050, **kwargs):
    input_dir = kwargs.get('input_dir', 'data/')
    output_dir = kwargs.get('output_dir', 'processed/')
    archive_dir = kwargs.get('archive_dir', 'data_archive/')
    try:
        in_filename = input_dir + fn_to_string(fn_tuple)
        out_filename = output_dir + fn_to_string(fn_tuple, with_ext=False)
        raw_audio = read_to_array(in_filename, fs=fs)
        tone_pallette = process_audio(raw_audio, fs=fs)
        np.save(out_filename, tone_pallette)
        move_when_done(in_filename, archive_dir)
        print time.ctime(), in_filename, ' -> ', out_filename
        return in_filename
    except ValueError as err:
        return err


if __name__ == '__main__':
    desc = '''
    Given a directory of raw audio (*.wav) files, extract tone-pallettes.
    '''
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('data_directory', type=str,
            help="Directory containing the raw audio files in .wav format")
    parser.add_argument('output_directory', type=str,
            help="Directory to place the resulting .npy binary files")
    parser.add_argument('--archive_directory', type=str, default='data_archive/',
            help='Directory to move raw files to once processed (for waypointing)')
    parser.add_argument('--num_workers', type=int, default=2,
            help="Limit the number of worker processes")
    args = parser.parse_args()
    main(args.data_directory, args.output_directory,
            args.archive_directory, num_workers=args.num_workers)
    main()
