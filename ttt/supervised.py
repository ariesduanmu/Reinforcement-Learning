import os
import random
import pickle
import tensorflow as tf

def save_network(session, tf_variables, file_path):
    variable_values = session.run(tf_variables)
    pickle.dump(variable_values, open(file_path,'wb'))

def load_network(session, tf_variables, file_path):
    variable_values = pickle.load(open(file_path, 'rb'))
    try:
        if len(variable_values) != len(tf_variables):
            raise ValueError("Network in file had different structure, variables in file: %s variables in memeory: %s"
                             % (len(variable_values), len(tf_variables)))
        for value, tf_variable in zip(variable_values, tf_variables):
            session.run(tf_variable.assign(value))
    except ValueError as ex:
        raise ValueError("""Tried to load network file %s with different architecture from the in memory network.
Error was %s
Either delete the network file to train a new network from scratch or change the in memory network to match that dimensions of the one in the file""" % (file_path, ex))




