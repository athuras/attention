# Given a config file, download, and convert to .wav the audio.
# example config file: echo "english 20\nfrench 15" > test.conf
from multiprocessing import Pool
import subprocess
import argparse


_ROOT_URL = 'http://accent.gmu.edu/soundtracks'
_ROOT_EXTENSION = '.mov'
# I really wanted to include this as a parameter for execute_work,
# but ultimately weird stuff happens with multiprocessing for some reason...
# I'm probably doing it wrong.
_DESTINATION = 'data'


def main(config_path, max_work=30, workers=1):
    config = gen_parse_file(config_path)
    scheduler = gen_work_units(config, max_work=max_work)
    pool = Pool(processes=workers)
    result = pool.map_async(execute_work, scheduler)
    print result.get()
    return


def form_filename(language, i, ext):
    return language + str(i) + ext


def download_and_convert(url_root, remote_fn, local_fn):
    '''applies subprocess.call to the ftp -> ffmpeg -> rm script.'''
    try:
        subprocess.call(['ftp', url_root + '/' + remote_fn, '.'])
        subprocess.call(['ffmpeg', '-i', remote_fn, local_fn])
        subprocess.call(['rm', remote_fn])
    except subprocess.CalledProcessError as e:
        return e
    finally:
        return True


def execute_work(work_unit):
    '''Iterate through the work unit, calling download_and_convert each time'''
    lang, start, inc = work_unit
    state = []
    for i in xrange(start, start + inc):
        remote_fn = form_filename(lang, i, _ROOT_EXTENSION)
        local_fn = _DESTINATION + '/' + form_filename(lang, i, '.wav')
        result = download_and_convert(url_root=_ROOT_URL,
                                      remote_fn=remote_fn,
                                      local_fn=local_fn)
        state.append((work_unit, result))
    return state


def gen_work_units(job_iter, max_work=30):
    '''Chunk big jobs into smaller, more manageable groups'''
    for lang, num in job_iter:
        start = 1
        while num > 0:
            inc = min(num, max_work)
            yield lang, start, inc
            num -= inc
            start += inc


def parse_file(path):
    return {lang: v for lang, v in gen_parse_file(path)}

def gen_parse_file(path):
    with open(path, 'r') as f:
        for l in f:
            items = l.rstrip('\n\t ').split(' ')
            yield items[0], int(items[1])


if __name__ == '__main__':
    desc = "Download the GMU Speech Accent Dataset (raw audio) and convert to .wav"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('config', type=str,
        help='Path to the config file (lang int) pairs')
    parser.add_argument('--max_work', type=int, default=10,
        help="limit on batch size")
    parser.add_argument('--num_workers', type=int, default=4,
        help="number of pooled external processes to spawn")
    args = parser.parse_args()
    main(args.config, args.max_work, args.num_workers)
