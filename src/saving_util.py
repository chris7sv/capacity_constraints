#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
from datetime import datetime

import numpy as np

"""
TODO:

Implement new file structure for experimental results

Suggestion 1:
    
    /name-of-experiment  ==> provided at running code, perhaps as a command
     prompt or a necessary argument; if exists, proceed
        /date and time
            /logs
            /model file
            /other files
            description file ==> contains all details of important parameters
            about the training of the model. Loading this file should basically
            replicate the experiment deterministically (might be necessary to
            include random generators)
        experiment description file - hand-typed. Outly the hypothesees being tested
        
=====================
this should be exported as a separate file and re-used as a module

"""

def create_subdirs(new_dir):
    """
    Creates subfolders for logs and model weights files
    """
    os.makedirs(os.path.join(new_dir, 'logs'))
    os.makedirs(os.path.join(new_dir, 'model'))
    return

def convert(d):
    """
    Converts numpy dtype of numbers to python base types to store in
    .json file. Otherwise, an exception is thrown for non-serializable
    objects.
    d - dict

    COMMENT: Does not return, the operations are performed in-place
    """
    for k, v in d.items():
        if isinstance(v, np.int64) or isinstance(v, np.int32):
            d[k] = int(v)
        elif isinstance(v, np.float32) or isinstance(v, np.float64):
            d[k] = float(v)
        else:
            continue

def export_hparams(history, save_dir: str, filename: str, more_params=None):
    """
    Pass hyperparameters from log file along with an optional dict with
    additional parameters to write to .json file

    history - keras history object, contains relevant model parameters to log

    save_dir - string, directory in which to save the file
    filename - string, the name of the .json file
    more_params - dict, contains additional parameters to be written to log
    """

    f = os.path.join(save_dir, filename)

    params = history.params
    lr = history.model.optimizer.get_config()
    params.update(lr)

    if more_params is not None:
        params.update(more_params)

    print(type(params))
    convert(params)

    json.dump(params, open(f, 'w'))
    return

def get_exp_name():
    """
    Input the name of the experiment, which will be used as the top-level
    directory to save trained models and log files.
    """
    exp_name = input("Please enter a name for the experiment:\t")
    return exp_name


def store_experiment(parent_dir, exp_name, sub_exp=None):
    if sub_exp is not None:
        new_dir = os.path.join(parent_dir, exp_name, sub_exp)
    else:
        new_dir = os.path.join(parent_dir, exp_name)
    if os.path.isdir(new_dir):
        subdir = str(datetime.now())[:-7]
        new_dir = os.path.join(new_dir, subdir)
        os.makedirs(new_dir)
        create_subdirs(new_dir)
   
    else:
        os.makedirs(new_dir)
        subdir = str(datetime.now())[:-7]
        new_dir = os.path.join(new_dir, subdir)
        os.makedirs(new_dir)
        create_subdirs(new_dir)

    return new_dir