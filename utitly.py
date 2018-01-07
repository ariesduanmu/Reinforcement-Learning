import os
import pickle

def save_dataset(filename, data):
    pickle.dump(data, open(filename, 'wb+'))

def read_dataset(filename):
    if os.path.isfile(filename):
        return pickle.load(open(filename, 'rb'))
    return None