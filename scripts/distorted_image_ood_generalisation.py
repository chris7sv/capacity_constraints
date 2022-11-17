#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
import glob
import json
import gc
import pickle
import sys

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
from skimage.color import rgb2gray, rgb2grey, gray2rgb

import tensorflow as tf

import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.datasets import cifar10

from tensorflow.keras.utils import to_categorical

sys.path.append('..')

from src.image_utils import * #iu
from src.saving_util import * #su
from src.models import *
from src.model_utils import *

"""
Created on Wed Oct  2 15:56:05 2019

@author: Chris Tsvetkov

Evaluate and compare performance on noisy images

"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = 'select mode: [e]valuate or [c]ompare'
    parser.add_argument('-m', '--mode', type=str, help=help_,
                        choices=['e','c'])
    help_ = 'pick a generalization test, <<sketch>> , <<noisy>> , <<high>> \
             or <<low>>.'
    parser.add_argument('-g','--gen', type=str, help=help_,
                        choices=['noisy','sketch', 'high','low',
                                 'ood-real','ood-draw','none'])
    help_ = 'Get top K accuracy for given k'
    parser.add_argument('--top', help=help_,type=int)
    help_ = 'pick level of noise in image (between 0 and 1)'
    parser.add_argument('-n', '--noise', type=float, help=help_,default=0.)
    help_ = 'Pick type of noise if selecting the noisy generalization test'
    parser.add_argument('--ntype', type=str, help=help_,
                        choices=['uniform', 'salt_pep'])
    parser.add_argument('-d','--data', type=str, choices =['train','test'])
    help_ = 'select name of first model'
    parser.add_argument('--model1', type=str,help=help_)
    help_ = 'select second model'
    parser.add_argument('--model2', type=str, help=help_)
    parser.add_argument('--standardize', action='store_true')

    args = parser.parse_args()

    set_session_params(tf.version.VERSION)
    
    """
    Checks if the experiments given exist
    """
    folders = [direc for direc in
               glob.glob(os.path.join(os.getcwd(), args.model1, '*'))
               if os.path.isdir(direc)]
    if len(folders) < 1:
        raise ValueError('invalid model1 argument, no such folder exists')
    if args.model2 is not None and args.mode == 'c':
        folders2 = glob.glob(os.path.join(os.getcwd(), args.model2, '*'))
        if len(folders2) < 1:
            raise ValueError('invalid model2 argument, no such folder exists')
    elif args.model2 is None and args.mode == 'c':
        raise ValueError('Please specify model2 for comparison')
    
    print(f'Using {args.data} data for evaluation.')

    if args.gen=='noisy':
        n_vals = np.round(np.arange(0, 0.51, 0.05) ,2)
    elif args.gen in ['high', 'low']:
        n_vals = np.arange(1, 10, 1)/2
    else:
        n_vals = [0]
    for n_val in n_vals:

        """
        Load data and pre-process
        """
        (x_train, y_train, y_train_copy), (x_test, y_test, y_test_copy) = load_and_prep_data(
            args.gen, args.ntype, n_val=n_val, standardize=True, save=False)

        """
        Runs each model in the experimental folder.

        Evaluates on the pre-loaded and pre-processed data
        Outputs performance.

        Terminates.
        """
        readouts = {}
        correct_topk = {}
        top_k = args.top
        for count, folder in enumerate(folders):

            #Load model and evaluate on data; store results in readouts
            K.clear_session()
            config_f = get_settings_from_file(os.path.join(folder, 'params.json'))
            model = build_model(model_t=config_f['model'],
                                bnorm=config_f['batch_norm'],
                                bneck=config_f['bottleneck'],
                                noise=config_f['noise'],
                                act=config_f['activation'])
            print(config_f['noise'])
            model.load_weights(
                os.path.join(folder, 'model/{mod}_cifar_10_classes_.h5'.
                             format(mod=args.model1[:3])))
            if args.data == 'train':
                readout = model.evaluate(x_train, y_train, verbose=0)
            else:
                readout = model.evaluate(x_test, y_test, verbose=0)
            print(readout)
            readout = [float(x) for x in readout]

            if config_f['noise'] is not None:
                curr_noise = round(config_f['noise'], 2)
            else:
                curr_noise = 0.0

            if curr_noise not in readouts.keys():
                readouts[curr_noise] = [readout]
            else:
                readouts[curr_noise].append(readout)

            # Performs a Top - K on model predictions on data, breaks down to correct/incorrect tirals and records.

            if args.top is not None:
                preds = model.predict(x_train if args.data == 'train' else x_test)
                in_top = (K.in_top_k(preds,
                                     y_train_copy if args.data == 'train' else y_test,
                                     tf.cast(top_k, tf.int32)))
                # print((np.unique(in_top,return_counts=True)))
                right = preds[in_top]
                wrong = preds[~in_top]
                top = preds[np.arange(preds.shape[0])[:, np.newaxis],
                            np.argsort(preds, axis=1)][:, -top_k:] #???
                top_right = right[np.arange(right.shape[0])[:, np.newaxis],
                                  np.argsort(right, axis=1)][:, -top_k:] #???
                top_wrong = wrong[np.arange(wrong.shape[0])[:, np.newaxis],
                                  np.argsort(wrong, axis=1)][:, -top_k:] #???
                if str(config_f['noise']) in correct_topk.keys():
                    correct_topk['{}'.format(config_f['noise'])].append(len(right))
                else:
                    correct_topk['{}'.format(config_f['noise'])] = [len(right)]
                print(correct_topk['{}'.format(config_f['noise'])],
                      len(correct_topk['{}'.format(config_f['noise'])]),
                      np.std(correct_topk['{}'.format(config_f['noise'])]))
                # print(f'Mean activation for top {top_k}: ',np.mean(top))
                # print(f'Mean activation for top {top_k} right: ', np.mean(top_right))
                # print(f'Mean activation for top {top_k} wrong: ', np.mean(top_wrong))

            # Visual progress output and some data clean-up
            if (count + 1) % 10 == 0:
                print(f'Finished: {count+1} folders')
            del model, config_f, readout
            gc.collect()
        # Save evaluate results for all models for specific dataset to file
        json.dump(readouts, open(os.path.join(os.getcwd(),
                                              'json files', 'updated',
                                              args.model1 + '_' + args.gen
                                              + '_{}_{}_{}.json'.
                                              format(args.ntype, n_val,
                                                     args.data)), 'w'))

        #Save top k results to file
        # mean_top_k = [('{}'.format(x),np.mean(y), np.std(y)) for x,y in readouts.items()]
        if args.top is not None:
            mean_top_k = {k:[np.mean(v), np.std(v)] for k, v in correct_topk.items()}
            mean_top_k_sorted = {}
            for key in sorted(mean_top_k):
                mean_top_k_sorted[key] = mean_top_k[key]
            print(mean_top_k_sorted)
            json.dump(mean_top_k_sorted,
                      open(os.path.join(os.getcwd(),
                                        'top_{k}_acc_{ntype}_{nval}_{model}_{data}.json'.
                                        format(k=top_k, ntype=args.ntype, nval=n_val,
                                               model=args.model1, data=args.data)), 'w'))
            json.dump(correct_topk,
                      open(os.path.join(os.getcwd(),
                                        'correct_top_{k}_{ntype}_{nval}_{model}_{data}.json'.
                                        format(k=top_k, ntype=args.ntype, nval=n_val,
                                               model=args.model1, data=args.data)), 'w'))
            pickle.dump(mean_top_k_sorted,
                        open(os.path.join(os.getcwd(),
                                          'top_{k}_acc_{ntype}_{nval}_{model}_{data}.p'.
                                          format(k=top_k, ntype=args.ntype, nval=n_val,
                                                 model = args.model1, data=args.data)), 'wb'))
            pickle.dump(correct_topk,
                        open(os.path.join(os.getcwd(),
                                          'correct_top_{k}_acc_{ntype}_{nval}_{model}_{data}.p'.
                                          format(k=top_k, ntype=args.ntype, nval=n_val,
                                                 model = args.model1, data=args.data)), 'wb'))
        del x_train, x_test, y_test, y_train
