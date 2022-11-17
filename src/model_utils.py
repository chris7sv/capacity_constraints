"""
@author: Chris Tsvetkov
"""

import json
import math
import glob
import os

import tensorflow as tf
import numpy as np

import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

from .models import *

def set_session_params(version):
    #Set session preferences
    try:
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:          
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        K.set_session(sess)



def vectorize_labels(labels, vec_size, max_vec_size=10):        
    #transform scalar labels to one-hot vectors of a given size
    
    new_labels = np.zeros([len(labels), vec_size], dtype='uint8')
    
    if vec_size < max_vec_size:        
        uniq = np.unique(labels)
        
        for idx, val in enumerate(uniq):
            labels[labels == val] = idx
  
    for i, label in enumerate(labels):
        # new_labels[i,labels[i]] = 1 # Refactored to use enumerate
        new_labels[i, label] = 1
    
    return new_labels

def get_settings_from_file(file):
    """
    Get settings for creating the model as recorded in

    a .json file.
    """

    with open(file, 'r') as f:
        settings = json.load(f)
    return settings


def build_model(model_t, lr=None ,bnorm = True, n_outputs=10, bneck=False,
        noise=None, drop=False, prec=None, act='relu', wd=0.0):

    """
    Create and compile model based on given parameters.

    This only creates a new model instance, weights are
    loaded separetely.

    model_t - type of model, vgg = VGG16, inc = small inception,
                alx = small alexnet.

    bnorm - batch normalization

    bneck - apply a bottleneck restricting the amount of filters
            in the first convolutional layer.

    noise - standard deviation of gaussian noise to be added to
            the activations of each convolutional layer.

    prec - CURRENTLY NOT FUNCTIONAL round activations to n decimal places.

    act - activation function to use.
    """

    if model_t == 'vgg':
        model = vgg_net(bneck, noise)
        opt = SGD(1e-2, 0.9, decay=1e-4)
    elif model_t == 'inc':
        model = inception_small(bnorm, n_outputs=n_outputs,
                                              bneck=bneck, noise=noise,
                                              prec=prec, act=act, wd=wd)
        if lr is None:
            opt = SGD(0.1, 0.9, decay=1e-4)
        else:
            opt = SGD(learning_rate=lr)
    elif model_t == 'alx':
        model = alexnet_small(bneck=bneck, n_outputs=n_outputs,
                                            noise=noise, drop=drop, prec=prec,
                                            act=act, wd=wd)
        if lr is None:
            opt = SGD(0.01, 0.9, decay=1e-4)
        else:
            opt = SGD(learning_rate=lr)
    elif model_t == 'dense':
        model = DenseNet(blocks=[6, 12, 24, 16], input_shape=(32, 32, 3),
                                       bottleneck=bneck, classes=n_outputs, noise=noise,
                                       activation=act)
        if lr is None:                               
            opt = SGD(1e-4, 0.9, decay=1e-4)
        else:
            opt = SGD(learning_rate=lr)
    elif model_t == 'mobile':
        model = MobileNet(input_shape=(32, 32, 3), dropout=False,
                                         classes=n_outputs, activation=act, noise=noise)
        if lr is None:
            opt = SGD(1e-4, 0.9, decay=1e-4)
        else:
            opt = SGD(learning_rate=lr)

            
    model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def step_decay(epoch):
    """
    Step decay function for keras LearningRateScheduler callback
    """
    initial_lrate = 0.01
    drop = 0.95
    epochs_drop = 1.0
    lrate = initial_lrate*math.pow(drop, math.floor((1 + epoch)/epochs_drop))
    return lrate

def sort_by_noise(model, select_level=False, noise=None):
    """
    Analysis helper function which sorts trained models in results folder by
    the standard deviation of internal noise used during training.

    Inputs:
    model - top-level dir which contains instances of a particular model
            simulation.
    select_level - boolean, whether to look for a particular level of noise
                   and select all models which use it, instead of sorting all
                   files. Defaults to False
    noise - if select_level is True, specify which level of noise to use. Leave
            at None if looking for models trained without noise
    
    Returns:
    Numpy array of subfolders in model directory sorted by noise levels
    """

    noise_vals = []
    noise_idx = []
    all_f = [direc for direc in glob.glob(os.path.join(os.getcwd(), model, '*')) if \
        os.path.isdir(direc)]
    for idx, f in enumerate(all_f):
        a = get_settings_from_file(os.path.join(f, 'params.json'))
        if select_level:
            if noise is not None:
                if a['noise'] is not None:
                    if np.round(a['noise'], 2) == np.round(noise, 2) or \
                         (np.abs(np.round(a['noise'], 2) - noise) < 1e-4):
                        noise_idx.append(idx)
            else:
                if a['noise'] is None:
                    noise_idx.append(idx)
        else:
            noise_vals.append(a['noise'])

    noise_vals = np.array(noise_vals)
    noise_vals[noise_vals == None] = 0.0

    if select_level:
        return np.array(all_f)[noise_idx]
    else:
        return np.array(all_f)[np.argsort(noise_vals)]
