import os
import gzip
import numpy as np
import random
import pescador

from tqdm import tqdm, tqdm_notebook

from util import *

        
def load(dataset="mdbsynth", lazy=True):
    """
    Returns a list of 3 elements [frequencies, raw, stft],
    by loading gzip-compressed npy files under directory specified in dataset.
    Each element of the tuple is a dictionary:
    {
        filename: (the numpy array),
        ...
    }
    
    Parameters
    ----------
    
    dataset : str
        the dataset subdirectory
    
    lazy : bool
        Instead of the array, each element in the dictionary becomes a lambda
        that returns the numpy array when called. Designed for memory heavy cases
    
    """
    
    class LazyFileLoader:
        def __init__(self, path):
            self.path = path

        def __call__(self, *args, **kwargs):
            with gzip.open(self.path) as f:
                return np.load(f)

    class LazyDirectoryLoader:
        def __init__(self, dataset, subdirectory, lazy):
            self.dataset = dataset
            self.subdirectory = subdirectory
            self.lazy = lazy
           
        def __call__(self, *args, **kwargs):
            dir = os.path.join(self.dataset, self.subdirectory)
            suffix = ".npy.gz"
            files = [f for f in os.listdir(dir) if f.endswith(suffix)]

            data = {}
            loop = files

            if not self.lazy:
                loop = is_notebook() and tqdm_notebook(loop) or tqdm(loop)
                loop.desc = "Loading %s data" % self.subdirectory

            for file in loop:
                path = os.path.join(dir, file)
                key = file.rstrip(suffix)

                data[key] = LazyFileLoader(path)
                if not lazy:
                    data[key] = data[key]()
                    
            return data
    
    results = []

    for subdirectory in ["frequencies", "raw", "stft"]:
        loader = LazyDirectoryLoader(dataset, subdirectory, lazy)
        results.append(loader)

    return results



def file_sampler(dataset, type, name, shuffle=True, **kwargs):
    """
    sampler function to be used by Pescador, corresponding to a file
    
    Parameters
    ----------
    
    type : str
        feature type, either 'raw' or 'stft'
    
    name : str
        name of the file, excluding the extension '.npy.gz'
    """
    index = {'raw': 1, 'stft': 2}[type]
    xs = dataset[index][name]  # shape=(M, N)
    ys = dataset[0][name]      # shape=(N) 
    
    if callable(xs):
        xs = xs()
    if callable(ys):
        ys = ys()
    
    assert(xs.shape[1] == ys.shape[0])
    
    indices = list(range(ys.shape[0]))
    if shuffle:
        random.shuffle(indices)
        
    while True:
        for i in indices:
            X = xs[:, i]
            y = ys[i]
            if y != 0:
                yield {'X': X, 'y': y}


def sequential_generator(dataset, type, names, batch_size):
    outputX = []
    outputY = []
    while True:
        for name in names:
            for sample in file_sampler(dataset, type, name, False):
                outputX.append(sample['X'])
                outputY.append(sample['y'])
                if len(outputY) == batch_size:
                    yield np.vstack(outputX), np.array(outputY)
                    outputX = []
                    outputY = []

                

def generator(type, shuffle=False, num_active_streams=230, batch_size=32, **kwargs):
    dataset = load(lazy=shuffle)
    
    # materialize the lazy loaders
    for i in [0, {'raw': 1, 'stft': 2}[type]]:
        dataset[i] = dataset[i]()
    
    names = list(dataset[0].keys())
    print(len(names))
    
    if not shuffle:
        return sequential_generator(dataset, type, names, batch_size)
    else:
        streamers = [pescador.Streamer(file_sampler, dataset, type, name) for name in names]
        mux = pescador.Mux(streamers, num_active_streams, **kwargs)
        batches = pescador.buffer_stream(mux, batch_size)
        return pescador.tuples(batches, 'X', 'y')


    
def stft_generator(**kwargs):
    return generator('stft', **kwargs)



def raw_generator(**kwargs):
    return generator('raw', **kwargs)
