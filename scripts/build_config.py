from collections import defaultdict
import argparse
import re
import subprocess


file_path = '~/Dropbox/speech_accent_archive\  materials/mov\ files'
_SPLITTER = re.compile('([a-zA-Z_]*)(\d*)(\.\D*)', re.IGNORECASE)


def main(dataset_dir, config_path):
    build_config_file(dataset_dir, config_path)
    return


def fn_parser(line):
    language, num, ext = apply_re(line, _SPLITTER)[0]
    try:
        num = int(num)
    except ValueError:
        return 'error', '0', '.err'
    return language, num, ext


def build_config_file(dataset_dir, config_path, fn_parser=fn_parser):
    '''Process the files in dataset_dir, and write to config_path'''
    acc = process_file_list(dataset_dir, fn_parser)
    write_config(config_path, acc)


def write_config(path, acc):
    '''Also strips out ugly variables'''
    flist = sorted(((k, v) for k, v in acc.iteritems()
                   if not (k == 'error' or len(k) == 1)))
    with open(path, 'wb') as f:
        for k, v in flist:
            f.write(k + ' ' + str(v) + '\n')
    return


def process_file_list(dir_path, parser):
    q = subprocess.check_output(['ls', dir_path])
    acc = defaultdict(int)
    for l in q.splitlines():
        term = parser(l)
        accumulator(term, acc)
    return acc


def accumulator(term, acc):
    '''Ignore extension for now'''
    lang, num, ext = term
    lang = str.lower(lang)
    lmax = acc[lang]
    acc[lang] = max(lmax, num)


def apply_re(line, _SPLITTER):
    '''apply _SPLITTER _sre.SRE_Pattern obj to line, call next() split outputs'''
    return _SPLITTER.findall(line)


if __name__ == '__main__':
    desc = 'Scrape the dropbox directory, and accumulate a config file for use with get_dataset.py'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('dataset_dir', type=str, help="Directory containing recordings")
    parser.add_argument('config_path', type=str, help="Location of output (config file)")
    args = parser.parse_args()
    main(args.dataset_dir, args.config_path)
