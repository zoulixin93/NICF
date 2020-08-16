#!/usr/bin/python
# encoding: utf-8

import numpy as np
import ipdb
import inspect
import random
import tensorflow as tf
import os
def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def get_objects(name_space):
    res = {}
    for name, obj in inspect.getmembers(name_space):
        if inspect.isclass(obj):
            res[name] = obj
    return res

def set_global_seeds(i):
    tf.set_random_seed(i)
    np.random.seed(i)
    random.seed(i)

def softmax(x):
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    softmax = numerator / denominator
    return softmax

def path_join(a,b):
    return os.path.join(a,b)

save4float = lambda x:str(round(x,4))

