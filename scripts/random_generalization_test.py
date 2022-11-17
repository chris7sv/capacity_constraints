#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import datetime
import os
import pickle


from tensorflow.keras.datasets import cifar10

#from tensorflow.keras.models import Model
#from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
#import skimage.io as io

from ..src.image_utils import * #iu
from ..src.saving_util import * #su
from ..src.models import *
from ..src.model_utils import *


"""
Code related to Appendix B from the paper. 


TODO:

Check whether changing any pixels produces the same change in output
"""

def shuffle_img(x_data):
    shape = x_data.shape
    x_data = x_data.reshape(shape[0],shape[1]*shape[2]*shape[3])
    for img in x_data:
      np.random.shuffle(img)
    x_data = x_data.reshape(shape)

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
def standardize_img(img):
    return ((img - np.mean(img))/(np.std(img)))


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

            if means is None:

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

def load_data(file):
    f = np.load(file)
    x_train = f['x_train']
    y_train = f['y_train']
    x_test = f['x_test']
    y_test = f['y_test']
    del f
    return x_train, y_train, x_test, y_test

def mod_pixels_and_eval(model, data, samples,pixels, pred_len):
    responses = np.ndarray(shape=(len(data), samples, pred_len))
    all_modded = np.ndarray(shape=(samples, 32*32*3))
    for idx,pic in enumerate(data):
        if idx % 10 == 0 and idx > 0:
            print(idx)
        for i in range(samples):
            pix = np.random.randint(0, len(data[0].flatten()), pixels)
            val = np.random.random(size=pixels)
            # val = np.random.normal(loc=np.mean(data), scale=np.std(data), size=pixels)
            img_copy = np.copy(pic).flatten()
            img_copy[pix] = val
            all_modded[i,:] = img_copy
        responses[idx,:,:] = model.predict([all_modded.reshape(-1,32,32,3)])
            
    return responses


#def mod_pixels_and_eval(model, data, samples,pixels, pred_len):
#    responses = np.ndarray(shape=(len(data), samples, pred_len))
#    
#    for idx,pic in enumerate(x_train[:100]):
#        pix = np.random.randint(0, len(x_train[0].flatten()), [samples,pixels])
#        val = np.random.randint(0,255,size=[samples,pixels])
#        img_copy = np.tile(pic.flatten(), samples).reshape(samples,-1)        
#        img_copy[np.arange(samples).reshape(-1,1), pix] = val
#        responses[idx,:,:] = model.predict([img_copy.reshape(samples,32,32,3)],batch_size=64)
#    print('done modding')        
#    return responses

def compare_performance(orig, modded, labels=None):
    """
        find the % matches between the orig classification and the modded classification
        find the level of confidence in orig, compare to mean level of confidence in correct and incorrect
        
    """
    class_acc = np.ndarray(shape=(len(orig),))
    confidence = np.ndarray(shape =(len(orig),4))
    if labels is not None:
        class_acc_labels = np.ndarray(shape=(len(labels)))
        # confidence_labels = np.ndarray(shape=(len(labels)))
    for num, pred in enumerate(orig):
        top_modded = np.argmax(modded[num,:,:], axis=1)
        p_correct = len(top_modded[(top_modded == np.argmax(pred))])/len(top_modded)
        p_correct_labels = len(top_modded[top_modded == np.argmax(labels[num])])/len(top_modded)
        class_acc[num] = p_correct
        class_acc_labels = p_correct_labels
        confidence[num, 0] = np.amax(pred)
#        print(np.mean(modded[num,:, np.argmax(pred)],axis=1))
        confidence[num, 1] = np.mean(modded[num,:,np.argmax(pred)])
        confidence[num, 2] = np.std(modded[num,:,np.argmax(pred)])
        confidence[num, 3] = np.median(modded[num,:,np.argmax(pred)])
#        confidence[num, 1] = np.mean(np.amax(modded[num,:,:], axis = 1))
        
    return class_acc,class_acc_labels, confidence
        


def plot_results(logs,data_type, num, mtype):
    
    path = 'logs/{mtype}_sanity_{m}_{n}_classes_{out}_{d}'.format(mtype=mtype,m=data_type, n = num,
                        out = 'in_output' if args.limit else '',
                        d = str(datetime.datetime.now())[:-7])
    
    print(logs.history.keys())
    plt.plot(logs.history['acc'])
    plt.plot(logs.history['val_acc'])# 1, 1, 1])
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.title('{m}_model accuracy'.format(m=data_type))
    plt.legend(['train','test'], loc = 'upper left')
    plt.savefig(path+'_acc.png')
    plt.show()
    plt.close()

    plt.plot(logs.history['loss'])
    plt.plot(logs.history['val_loss'])
    plt.title('{m}_model_loss'.format(m=data_type))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','test'], loc = 'upper left')
    
    plt.savefig(path+'_loss.png')
    plt.show()
    plt.close()

class pick_output(Callback):
    def __init__(self, eval_data, layer_n, true_label):
        self.eval_data = eval_data
        self.true_label = true_label
        self.layer_n = layer_n
    def on_epoch_end(self, batch, logs = {}):
        get_output = K.function([self.model.layers[0].input],[self.model.layers[self.layer_n].output])
        pred = get_output([self.eval_data])[0]
        print(len(self.model.layers))
        print('Pred: ',pred)
        print('True: ', self.true_label)
        print(len(np.nonzero(pred)[0]))

    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    help_ = "Select mode: train or eval"
    parser.add_argument("--mode", help = help_, choices = ['train','eval'])
    help_ = "Choose between training model on CIFAR10 or random data - gauss, shuffle or label"
    parser.add_argument('-d','--data', choices=['rnd-gauss','rnd-labels'])
    help_ = "Choose model architecture. Default : inception"
    parser.add_argument('-m', '--model', help = help_, choices = ['alx','vgg','inc'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default = 1e-4)
    parser.add_argument('--batch_size', type=int, default = 128)
    help_ = 'Specify number of classes to train on.'
    parser.add_argument('-n', '--n_classes', type = int, help = help_, default = 10)
    help_ = 'Limit model output layer dimension to specified n classes'
    parser.add_argument('--limit',help = help_ , action =  "store_true")
    help_= "Specify weights file to load"
    parser.add_argument('-w','--weights', help=help_)
    help_ = "Generate new random dataset"
    parser.add_argument("--new", help=help_, action = "store_true")
    parser.add_argument("--path", type=str)
    parser.add_argument("--standardize", action="store_true")
    parser.add_argument("--bneck", action="store_true")
    parser.add_argument("--noise",type=float,default=None)
    parser.add_argument("--act", type=str, choices=['relu','sig'], default='relu')
    
    args = parser.parse_args()
    
    num_classes = 10 # max number of classes
    n = args.n_classes
    

    print(f'Training on {n} classes')
    filedir = 'saved_models/'
    
    if args.data is None:
        raise NameError('Please select an option for training. See --help for more.')
    

    
    means = [125.3, 123.0, 113.9]#[x/255.0 for x in [125.3, 123.0, 113.9]]
    std = [63.0, 62.1, 66.7]#[x / 255.0 for x in [63.0, 62.1, 66.7]]
    
    
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train/255.
    x_test = x_test/255.

    # if args.standardize:
    #     x_train = standardize_data(x_train)
    #     x_test = standardize_data(x_test)   
    
    if n != 10:
        sample = np.random.choice(range(num_classes),n,replace=False)#np.array([7,9])
        sample_idx = np.isin(y_train,sample)
        sample_idx_test = np.isin(y_test,sample)
        x_train, y_train = (x_train[sample_idx.flatten()],
                                                  y_train[sample_idx])
    
        x_test, y_test = (x_test[sample_idx_test.flatten()],
                                               y_test[sample_idx_test])
        print(sample)
        
    
    if args.limit:
        num_classes = n
    
    y_train, y_test = vectorize_labels(y_train,num_classes), vectorize_labels(y_test, num_classes)

    print('max, min: ',np.amax(x_train), np.amin(x_train))
    
    print(x_train.shape,'\n',x_test.shape)
    
    if args.data == 'rnd-gauss':
        
        if args.new:
            x_train = generate_random_image(len(x_train),x_train.shape[1:],means=means,std=std)
            print(x_train.shape)
            np.random.shuffle(y_train)
            x_test = generate_random_image(len(x_test),x_test.shape[1:])
            np.random.shuffle(y_test)
            datafile = os.path.join(save_dir,'data','{d}'.format(d=str(datetime.datetime.now())[:-7]))
            np.savez(datafile, x_train=x_train,x_test=x_test, y_train = y_train, y_test = y_test)
        else:
            if args.path is None:
                raise NameError("Either provide a path for data using --path, or generate new stimuli using --new.")
            else:  
                x_train, y_train, x_test, y_test = load_data(args.path)
        if args.standardize:
            x_train = standardize_data(x_train)
            x_test = standardize_data(x_test)
        
        print('max, min: ',np.amax(x_train), np.amin(x_train))


        print(x_train.shape,'\n',x_test.shape)
        
        print(y_train[:10],y_test[:10])
        
    elif args.data == 'rnd-labels':
        
        np.random.shuffle(y_train)
        np.random.shuffle(y_test)

#Set session preferences
       
    set_session_params(tf.version.VERSION)
        
#Set some hyperparameters and build model
#   
#train model; include commented lines to monitor training closer each epoch
    
    if args.mode == 'train':
        model = build_model(args.model, bnorm=True, n_outputs = args.n_classes, bneck = args.bneck, noise=args.noise, act=args.act)   

        out_tracker = pick_output(x_train[:2], 29, y_train[:2])   
    
        logs = model.fit(x_train, y_train,
                  batch_size=args.batch_size,
                  epochs=args.epochs,
                  validation_data=(x_test,y_test),
                  shuffle=True)#,
    #              callbacks = [out_tracker])
            
            
        model.save_weights(filedir+'{mtype}_{m}_{n}_classes_{out}_lr_{lr}_epochs_{e}_{date}.h5'.format(mtype = args.model,
                           m = args.data, n = n,
                           out = 'in_output' if args.limit else '',
                           lr = args.lr,e = args.epochs, date=str(datetime.datetime.now())[:-7]))
        logs.model = None
        pickle.dump(logs,open('logs/{mtype}_logs_sanity_{m}_{n}_classes_out_{d}.p'.format(mtype = args.model,m=args.data, n = n,
                              out = 'in_output' if args.limit else '',
                              d=str(datetime.datetime.now())[:-7]), 'wb'))
        # plot_results(logs, args.data, n, args.model)
        
    elif args.mode == 'eval':
        model = build_model(args.model, bnorm=True, n_outputs = args.n_classes, bneck = args.bneck, noise=args.noise, act=args.act)   

        #pick a random pixel(s) to distort in some images
        model.load_weights(args.weights)

        p = 0.10
        n = int(x_train[0].size*p)
        print(f'Modifying {p*100}% of image pixels, or {n} total')
        samples = 1000
        draw = np.random.randint(0,len(x_train),100)
        orig = model.predict(x_train[draw])
        pred_len = orig.shape[-1]
        top_orig = np.argmax(orig,axis=1)
        print(np.argmax(orig[0]))
        print(model.evaluate(x_train[draw],y_train[draw]))
        modded = mod_pixels_and_eval(model, x_train[draw], samples, n, pred_len)

        # if args.standardize:
        #     x_train = standardize_data(x_train)
        #     x_test = standardize_data(x_test)
        
        acc, acc_labels, conf = compare_performance(orig, modded, y_train[draw])

        print(np.mean(acc),np.mean(acc_labels),np.mean(conf[0]))
        
        np.savez(f'results/mod_pixels_{p}_{samples}_samples.npz', orig=orig,acc=acc,conf=conf)