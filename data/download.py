"""
Downloads the following:
- Glove vectors
- Stanford Natural Language Inference (SNLI) Corpus

"""

import sys
import os
import zipfile
import gzip

def download(url, dirpath):
    filename = url.split('/')[-1]
    filepath = os.path.join(dirpath, filename)
    os.system('wget {} -O {}'.format(url, filepath))
    return filepath

def unzip(filepath):
    print("Extracting: " + filepath)
    dirpath = os.path.dirname(filepath)
    with zipfile.ZipFile(filepath) as zf:
        zf.extractall(dirpath)
    os.remove(filepath)

def download_wordvecs(dirpath):
    if os.path.exists(dirpath):
        print('Found Glove vectors - skip')
        return
    else:
        os.makedirs(dirpath)
    url = 'http://www-nlp.stanford.edu/data/glove.840B.300d.zip'
    unzip(download(url, dirpath))

def download_snli(dirpath):
    if os.path.exists(dirpath):
        print('Found SNLI dataset - skip')
        return
    else:
        os.makedirs(dirpath)
    url = 'https://nlp.stanford.edu/projects/snli/snli_1.0.zip'
    unzip(download(url, dirpath))


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.realpath(__file__))
    snli_dir = os.path.join(base_dir, 'snli')
    wordvec_dir = os.path.join(base_dir, 'glove')
    download_snli(snli_dir)
    download_wordvecs(wordvec_dir)

