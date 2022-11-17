#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 14:55:44 2019

@author: Chris Tsvetkov
"""

"""

A sanity check test to replicate Sammy Bengio's results with Cifar

"""

import argparse
import os 
import pickle
import gc
import math
import sys


from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import SGD#,adam, adagrad, rmsprop
from tensorflow.keras.callbacks import Callback, LearningRateScheduler
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
import tensorflow as tf

import numpy as np
#import skimage.io as io
import matplotlib.pyplot as plt

sys.path.append('..')

# from src.image_utils import * #iu
from src.saving_util import * #su
from src.models import *
from src.model_utils import *

K.clear_session()
#K.set_floatx('float16')
#K.set_epsilon(1e-4)

def shuffle_img(x_data):
    shape = x_data.shape
    x_data = x_data.reshape(shape[0],shape[1]*shape[2]*shape[3])
    for img in x_data:
      np.random.shuffle(img)
    x_data = x_data.reshape(shape)

    
def standardize_img(img):
    return (img - np.mean(img))/np.std(img)
#
#def per_channel_standardize(img):
#    for i in range(3):
#        img[:,:,i] = n

def standardize_data(data):
    for i in range(len(data)):
        data[i] = standardize_img(data[i])
    return data

def clip_extremes(data):
    data[data<0] = 0
    data[data>255] = 255
    return data

def generate_random_image(num_samples,data_shape,random_type='rnd-gauss', means = None, std = None):
    
    data = np.ndarray([num_samples, data_shape[0],data_shape[1],data_shape[2]])
    if random_type == 'rnd-gauss':
        for i in range(num_samples):
#            data_point = np.ndarray(shape=data_shape)
#            sampl =  np.ndarray(shape = (data_shape[0]*data_shape[1]*data_shape[2]))
            if means is None:
#                for j in range(len(sampl)):
#                    sampl[j] = np.random.normal(120.75, 69.93,1)
#                    data[i] = sampl.reshape((data_shape[0],data_shape[1],data_shape[2]))

                data[i] = np.random.normal(120.75, 69.93,
                                  (data_shape[0]*data_shape[1]*data_shape[2])).reshape(
                                          data_shape[0],data_shape[1],data_shape[2])
            elif type(means) in [list, tuple]:
                data_point = np.ndarray(shape=data_shape)
                for idx,stat in enumerate(zip(means, std)):
                    data_point[:,:,idx] = np.random.normal(stat[0] , stat[1],
                                      (data_shape[0]*data_shape[1])).reshape(
                                              data_shape[0],data_shape[1])
                data[i] = np.copy(data_point)
    else:
        raise NameError('Not implemented yet, please use rnd-gauss noise')
    
#    data = clip_extremes(data)
    return data/255.

def vectorize_labels(labels, vec_size, max_vec_size=10):        
    #transform scalar labels to one-hot vectors of a given size
    
    new_labels = np.zeros([len(labels),vec_size],dtype='uint8')
    
    if vec_size < max_vec_size:        
        uniq = np.unique(labels)
        
        for idx, val in enumerate(uniq):
            labels[labels==val] = idx
  
    for i in range(len(labels)):
        new_labels[i,labels[i]] = 1
    
    return new_labels

def plot_results(logs,data_type, num, model_type,savedir):
    
    path = os.path.join(save_dir,'logs','{mtype}_sanity_{m}_{n}_classes_{out}'.format(mtype=model_type,m=data_type, n = num,
                        out = 'in_output' if args.limit else ''))    
    print(logs.history.keys())
    fig = plt.figure(clear=True)
    plt.plot(logs.history['accuracy' if tf.version.VERSION in ['2.0.0-rc0','2.2.0','2.1.0'] else 'acc'])
    plt.plot(logs.history['val_accuracy' if tf.version.VERSION in ['2.0.0-rc0','2.2.0','2.1.0'] else 'val_acc'])# 1, 1, 1])
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.title('{m}_model accuracy'.format(m=data_type))
    plt.legend(['train','test'], loc = 'upper left')
    plt.savefig(path+'_acc.png')
    plt.close(fig)
#    plt.show()
    
    fig = plt.figure(clear=True)
    plt.plot(logs.history['loss'])
    plt.plot(logs.history['val_loss'])
    plt.title('{m}_model_loss'.format(m=data_type))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','test'], loc = 'upper left')
    
    plt.savefig(path+'_loss.png')
    plt.close(fig)
#    plt.show()


class pick_output(Callback):
    def __init__(self, eval_data, layer_n, true_label):
        self.eval_data = eval_data
        self.true_label = true_label
        self.layer_n = layer_n
    def on_epoch_end(self, batch, logs = {}):
        get_output = K.function([self.model.layers[0].input],[self.model.layers[self.layer_n].output])
        pred = get_output([self.eval_data])[0]
        print(self.model.layers[self.layer_n].name)
        #print(len(self.model.layers))
        print('Pred: ',pred)
        print('True: ', self.true_label)
        print(len(np.nonzero(pred)[0]))
        
        
# def set_session_params(version):
#     #Set session preferences
#     if version == '2.0.0-rc0':
#         physical_devices = tf.config.experimental.list_physical_devices('GPU')
#         assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
#         tf.config.experimental.set_memory_growth(physical_devices[0], True)
#     else:           
#         config = tf.ConfigProto()
#         config.gpu_options.allow_growth = True
#         sess = tf.Session(config=config)
#         K.set_session(sess)   
        

def step_decay(epoch):
    initial_lrate = 0.01#0.1
    drop = 0.95#0.5
    epochs_drop = 1.0#15
    lrate = initial_lrate * math.pow(drop,  
           math.floor((1+epoch)/epochs_drop))
    return lrate

          

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    help_ = "Choose between training model on CIFAR10 or random data - gauss, shuffle or label"
    parser.add_argument('-d', '--data', choices=['cifar', 'rnd-gauss', 'rnd-shuffle', 'rnd-labels'])
    help_ = "Choose model architecture. Default : inception"
    parser.add_argument('-m', '--model', help=help_, choices=['vgg', 'inc', 'alx', 'dense', 'mobile'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=128)
    help_ = 'Specify number of classes to train on.'
    parser.add_argument('-n', '--n_classes', type=int, help=help_, default=10)
    help_ = 'Limit model output layer dimension to specified n classes'
    parser.add_argument('--limit', help=help_ , action="store_true")
    help_ = "Resume training with specified weights"
    parser.add_argument('-r', '--resume', help=help_)
    help_ = "Standardize data"
    parser.add_argument('-s', '--standardize', help=help_, action="store_true")
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('-b', '--bottleneck', action="store_true")
    parser.add_argument('--noise', type=float, default=0.0)
    help_ = "Select an activation function, relu or sigmoid"
    parser.add_argument("-a", "--act", type=str, help=help_, choices=['relu', 'sigmoid'], default='relu')
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--prec", type=str, choices=['float16', 'float32'])
    parser.add_argument("--noise_min", type=float, required=False, help='Minimum value for noise range')
    parser.add_argument("--noise_max", type=float, required=False, help='Maximum value for noise range')
    parser.add_argument("--noise_step", type=float, required=False, help='Increment size for noise range sweep')
    
    args = parser.parse_args()
    
    set_session_params(tf.version.VERSION)
    
    #Set some hyperparameters and build model

    num_classes = 10 # max number of classes
    n = args.n_classes
    

    print(f'Training on {n} classes')
    
    
    exp_name = get_exp_name()
    
    if args.data is None:
        raise NameError('Please select an option for training. See --help for more.')
    
# Pre-computed mean and std per channel for entire cifar dataset
    
    means = [125.3, 123.0, 113.9]#[x/255.0 for x in [125.3, 123.0, 113.9]]
    std = [63.0, 62.1, 66.7]#[x / 255.0 for x in [63.0, 62.1, 66.7]]
    
# Load data and scale    
    
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train/255.
    x_test = x_test/255.
    
    
# If working with less than 10 categories, create a random sample 
    
    if n != 10:
        sample = np.random.choice(range(num_classes), n, replace=False)
        sample_idx = np.isin(y_train, sample)
        sample_idx_test = np.isin(y_test, sample)
        x_train, y_train = (x_train[sample_idx.flatten()],
                                                  y_train[sample_idx])
    
        x_test, y_test = (x_test[sample_idx_test.flatten()],
                                               y_test[sample_idx_test])
        print(sample)

# Limit the output units of the model to match the number of categories in the sample        
    
    if args.limit:
        num_classes = n
    
    y_train = vectorize_labels(y_train, num_classes)
    y_test = vectorize_labels(y_test, num_classes)
    
#Standardize the data
    
    if args.standardize:
        x_train = standardize_data(x_train)
        x_test = standardize_data(x_test)
        
# Printing check    
    print('max, min: ',np.amax(x_train), np.amin(x_train))
    print('mean, std: ', np.mean(x_train), np.std(x_train))    
    print(x_train.shape, '\n',x_test.shape)

    if args.noise_min:
        cur_noise = args.noise_min
    else:
        cur_noise = 0.0

    if args.noise_max:
        max_noise = args.noise_max
    else:
        max_noise = (0.2 + 1e-5) if args.act=='sigmoid' else (1.2 + 1e-5) 

    if args.noise_step:
        noise_step = args.noise_step
    else:
        noise_step = 0.02 if args.act=='sigmoid' else 0.1 
    
    for rep in range(args.repeat):
        if args.noise_min:
            cur_noise = args.noise_min
        else:
            cur_noise = 0.0
        while cur_noise <= max_noise:
            
            print(cur_noise)
            save_dir = store_experiment(os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir())), 'data'), exp_name)

            K.clear_session()
    #        tf.config.optimizer.set_jit(True)


            if args.data == 'rnd-gauss':

        #Generate new synthetic data from a gaussian distribution. Shuffling labels is unnecessary, but added for comfort ^^

                x_train = generate_random_image(len(x_train),x_train.shape[1:],means=means,std=std)
                print(x_train.shape)
                np.random.shuffle(y_train)
                x_test = generate_random_image(len(x_test),x_test.shape[1:])
                np.random.shuffle(y_test)

        #Standardize data        

                if args.standardize:
                    x_train = standardize_data(x_train)
                    x_test = standardize_data(x_test)

        #Printing check        
                print('max, min: ',np.amax(x_train), np.amin(x_train))
                print('mean, std: ', np.mean(x_train), np.std(x_train))    
                print(x_train.shape,'\n',x_test.shape)


            elif args.data == 'rnd-shuffle':    
            #Shuffle the pixels in each image. Currently not working quite as expected.        
                raise NameError('Not functional presently.')

                shuffle_img(x_train)
                shuffle_img(x_test)

            elif args.data == 'rnd-labels':            
            #Shuffle all labels

                np.random.shuffle(y_train)
                np.random.shuffle(y_test)

            elif args.data == 'cifar':            
                pass        

            bnorm = True
            bneck = args.bottleneck
            noise = cur_noise if cur_noise > 0 else None
            prec = args.prec 
            act = args.act
            lr_decay = 0
            
            print(noise)
            ##########################
            #Deprecated, moved to model_utils
            ##########################
            # if args.model is None:
            #     model_type = 'inc'
            #     print('No architecture provided, using default settings.')
            # else:
            #     model_type = args.model
            # print(f'Using {model_type} as the model architecture. Activation function: {act}')
            # if model_type == 'vgg':
            #     model = model_test_v2.vgg_net(num_classes, bneck=bneck, noise=noise, prec=prec)
            #     opt = SGD(1e-2, 0.9, decay = lr_decay)

            # elif model_type == 'inc':
            #     model = model_test_v2.inception_small(bnorm,num_classes, bneck = bneck,noise=noise, prec=prec, act=act)
            #     opt = SGD(0.1, 0.9, decay = lr_decay)
            # elif model_type == 'alx':
            #     model = model_test_v2.alexnet_small(num_classes, bneck, noise, prec=prec, act=act)
            #     opt = SGD(0.01, 0.9, decay = lr_decay)
            # model.compile(opt,loss='categorical_crossentropy', metrics = ['accuracy'])
            
            model_type = args.model
            model = build_model(model_type, n_outputs=n, bneck=bneck, noise=noise, act=act)

            # model.summary()

            # plot_model(model, to_file=os.path.join(save_dir,'model_plot.png'), show_shapes=True, show_layer_names=True)

    #train model; include commented lines to monitor training closer each epoch

            if args.resume is not None:
                model.load_weights(args.resume)

            lrate = LearningRateScheduler(step_decay, verbose=1)

            out_tracker = pick_output(x_train[:10], 29, y_train[:10])   

            logs = model.fit(x_train, y_train,
                      batch_size=args.batch_size,
                      epochs=args.epochs,
                      validation_data=(x_test,y_test),
                      verbose=args.verbose,
                      shuffle=True,
                      callbacks = [lrate])#,out_tracker])

            export_hparams(logs,save_dir,'params.json',{'data':args.data,'model':args.model,'batch_norm':bnorm,
                                                           'bottleneck':bneck,'noise':noise, 'standardized':args.standardize,
                                                           'activation':args.act, 'precision':args.prec})

            model.save_weights(os.path.join(save_dir,'model','{mtype}_{m}_{n}_classes_{out}.h5'.format(mtype = model_type,
                               m = args.data, n = n,
                               out = 'in_output' if args.limit else '')))
            logs.model = None

            pickle.dump(logs,open(os.path.join(save_dir,'logs','{mtype}_logs_sanity_{m}_{n}_classes_out.p'.format(mtype = model_type,m=args.data, n = n,
                                  out = 'in_output' if args.limit else '')),'wb'))
            plot_results(logs, args.data, n, model_type, save_dir)
            
            del model
            gc.collect()
            
            cur_noise += noise_step
            # cur_noise += 0.02 if args.act=='sigmoid' else 0.1 
        
