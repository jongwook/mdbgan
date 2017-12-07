import os
import gzip
import numpy as np
import random

from tqdm import tqdm, tqdm_notebook

from util import *

class lazyloader:
    def __init__(self, path):
        self.path = path

    def __call__(self, *args, **kwargs):
        with gzip.open(self.path) as f:
            return np.load(f)


def load(dataset="mdbsynth", lazy=True, verbose=False):
    """
    Returns a tuple of 3 elements (frequencies, raw, stft),
    by loading gzip-compressed npy files under directory specified in dataset.
    Each element of the tuple is a dictionary,
    {
        filename: (the numpy array),
        ...
    }
    
    :param dataset
    
        the dataset subdirectory
    
    :param lazy 
        
        Instead of the array, each element in the dictionary becomes a lambda
        that returns the numpy array when called. Designed for memory heavy cases
    """

    results = []

    for subdirectory in ["frequencies", "raw", "stft"]:
        suffix = ".npy.gz"
        dir = os.path.join(dataset, subdirectory)
        files = [f for f in os.listdir(dir) if f.endswith(suffix)]

        data = {}
        loop = files

        if verbose:
            loop = is_notebook() and tqdm_notebook(loop) or tqdm(loop)
            loop.desc = "Loading %s data" % subdirectory

        for file in loop:
            path = os.path.join(dir, file)
            key = file.rstrip(suffix)

            data[key] = lazyloader(path)
            if not lazy:
                data[key] = data[key]()

        results.append(data)

    return results


def generator(type, dataset="mdbsynth", lazy=True, verbose=False):
    data = load(dataset, lazy, verbose)
    # TODO


def stft_generator():
    pass

def raw_generator():
    pass
