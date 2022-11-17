import argparse
import os 
import pickle
import gc
import math
import sys

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD#,adam, adagrad, rmsprop
from tensorflow.keras.callbacks import Callback, LearningRateScheduler
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K

#import skimage.io as io

sys.path.append('..')

from src.image_utils import * #iu
from src.saving_util import * #su
from src.models import *
from src.model_utils import *

"""
Capacity constraint simulations from Section 3.3, procedure for random data

Code by Chris Tsvetkov
"""

def standardize_img(img):
    return (img - np.mean(img))/np.std(img)

def standardize_data(data):
    for i in range(len(data)):
        data[i] = standardize_img(data[i])
    return data

def clip_extremes(data):
    data[data < 0] = 0
    data[data > 255] = 255
    return data

def generate_random_image(num_samples, data_shape, random_type='rnd-gauss',
        means=None, std=None):
    
    data = np.ndarray([num_samples, data_shape[0], data_shape[1], data_shape[2]])
    if random_type == 'rnd-gauss':
        for i in range(num_samples):
            if means is None:
                data[i] = np.random.normal(120.75, 69.93,
                                           (data_shape[0]
                                            *data_shape[1]
                                            *data_shape[2])).reshape(
                                                data_shape[0],
                                                data_shape[1],
                                                data_shape[2])
            elif isinstance(means, (list, tuple)):
                data_point = np.ndarray(shape=data_shape)
                for idx, stat in enumerate(zip(means, std)):
                    data_point[:, :, idx] = np.random.normal(stat[0], stat[1],
                                                             (data_shape[0]
                                                              *data_shape[1])
                                                            ).reshape(
                                                                data_shape[0],
                                                                data_shape[1])
                data[i] = np.copy(data_point)
    else:
        raise NameError('Not implemented yet, please use rnd-gauss noise')

    return data/255.


def plot_results(logs, data_type, num, model_type, save_dir):

    path = os.path.join(save_dir, 'logs',
                        '{mtype}_sanity_{m}_{n}_classes_{out}'.format(
                            mtype=model_type, m=data_type, n=num,
                            out='in_output' if args.limit else ''))

    print(logs.history.keys())

    fig = plt.figure(clear=True)
    try:
        plt.plot(logs.history['accuracy'])
    except:
        plt.plot(logs.history['acc'])
    try:
        plt.plot(logs.history['val_accuracy'])
    except:
        plt.plot(logs.history['val_acc'])

    plt.xlabel('epoch')
    plt.title('{m}_model accuracy'.format(m=data_type))
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(path + '_acc.png')
    plt.close(fig)
    
    fig = plt.figure(clear=True)
    plt.plot(logs.history['loss'])
    plt.plot(logs.history['val_loss'])
    plt.title('{m}_model_loss'.format(m=data_type))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    
    plt.savefig(path + '_loss.png')
    plt.close(fig)

class PickOutput(Callback):

    def __init__(self, eval_data, layer_n, true_label):
        self.eval_data = eval_data
        self.true_label = true_label
        self.layer_n = layer_n

    def on_epoch_end(self, batch, logs={}):
        get_output = K.function([self.model.layers[0].input],
                                [self.model.layers[self.layer_n].output])
        pred = get_output([self.eval_data])[0]
        print(len(self.model.layers))
        print('Pred: ', pred)
        print('True: ', self.true_label)
        print(len(np.nonzero(pred)[0]))        

class GetLastLR(Callback):
    def __init__(self):
        lr = None
    def on_epoch_end(self, epoch, logs=None):
        lr = K.eval(self.model.optimizer.lr)
        print("Learning rate:", lr)       

def step_decay(epoch):
    initial_lrate = 0.01
    drop = 0.95
    epochs_drop = 1.0
    lrate = initial_lrate*math.pow(drop, math.floor((1 + epoch)/epochs_drop))
    return lrate


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # help_ = "Choose between training model on CIFAR10 or random data\
    #  - gauss, shuffle or label"
    # parser.add_argument('-d', '--data',
    #                     choices=['cifar', 'rnd-gauss', 'rnd-shuffle',
    #                              'rnd-labels'])
    help_ = "Choose model architecture. Default : inception"
    parser.add_argument('-m', '--model', help=help_, choices=['vgg', 'inc',
                                                              'alx'])
    parser.add_argument('-d', '--data', choices=['cifar', 'rnd-labels'],
                        default='cifar')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=128)
    help_ = 'Specify number of classes to train on.'
    parser.add_argument('-n', '--n_classes', type=int, help=help_, default=10)
    help_ = 'Number of examples per category'
    parser.add_argument('-s', '--stimuli', type=int, help=help_, default=1000)
    help_ = 'Limit model output layer dimension to specified n classes'
    parser.add_argument('--limit', help=help_, action="store_true")
    help_ = "Standardize data"
    parser.add_argument('--standardize', help=help_, action="store_true")
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('-b', '--bottleneck', action="store_true")
    parser.add_argument('--noise', type=float, default=0.0)
    help_ = "Select an activation function, relu or sigmoid"
    parser.add_argument("-a", "--act", type=str, help=help_,
                        choices=['relu', 'sigmoid'], default='relu')

    K.clear_session()
    
    args = parser.parse_args() 
    
    rnd= np.random.RandomState()

           
    set_session_params(tf.version.VERSION)
    
    #Set some hyperparameters and build model
    set_of_x = [10, 100]#10000,10,100,
    set_of_n = np.array([10])
    #np.concatenate((np.arange(2,10,1),np.array([10,50,100,500,1000])))
    #  # still have to finish doing n=50, x=1000 !!!!!!!!!!!
    # n = args.n_classes
    # x = args.stimuli
    means = [125.3, 123.0, 113.9]#[x/255.0 for x in [125.3, 123.0, 113.9]]
    std = [63.0, 62.1, 66.7]#[x / 255.0 for x in [63.0, 62.1, 66.7]]
    # print(f'Training on {x} examples in {n} classes, X = {x*n}')

    exp_name = su.get_exp_name()
    for n in set_of_n:
        for x in set_of_x:
            exp_name_ii = os.path.join(exp_name, f'n_{n}_x_{x}')
            print(f'Training on {x} examples in {n} classes, X = {x*n}')
            for rep in range(args.repeat):
                print(f'Iteration {rep} out of {args.repeat}')
                max_noise = (.7 + 1e-5) if args.act == 'relu' else (0.5 + 1e-5)
                #(2.0 + 1e-5) if args.act=='relu' else (0.5 + 1e-5)#(.7 + 1e-5)\
                #  if args.act=='relu' else (0.5 + 1e-5)

                cur_noise = 0.#.6
                # cur_noise = 1. if (args.model == 'alx' and \
                #   args.act=='relu') else .5 if (args.model == 'inc' \
                #   and args.act=='relu') \
                #   else .2 if (args.model=='alx' and args.act == 'sigmoid')\
                #   else .14
                while cur_noise <= max_noise:
                    
                    print(cur_noise)
                    save_dir = store_experiment(os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir())), 'data'), exp_name_ii)

                    K.clear_session()

                    #Generate some labels 
                    y_train = np.stack((np.arange(0, n, 1),)*x)
                    y_test = np.stack((np.arange(0, n, 1),)*int(x/10))
                    y_train = y_train.flatten()
                    y_test = y_test.flatten()

                    # y_train = vectorize_labels(y_train,10)
                    # y_test = vectorize_labels(y_test,10)
                    y_train = to_categorical(y_train)
                    y_test = to_categorical(y_test)
                    np.random.shuffle(y_train)
                    np.random.shuffle(y_test)

                    #Generate the images

                    x_train = iu.generate_random_image(n*x, (32, 32, 3),
                                                       means=means, std=std,
                                                       rnd=rnd)
                    x_test = iu.generate_random_image(int(n*x/10), (32, 32, 3),
                                                      means=means, std=std,
                                                      rnd=rnd)

                    if args.standardize:
                        x_train = standardize_data(x_train)
                        x_test = standardize_data(x_test)

                    print(x_train.shape, y_train.shape)

                    all_steps_full = 5000//args.batch_size*100
                    epochs = min(1000, all_steps_full//max(
                        1, ((n*x)//args.batch_size)))
                    print(f'Training for {epochs} epochs')

                    initial_lr = 0.01 if args.model == 'alx' else 0.1
                    steps = len(x_train)#epochs*10
                    decay_rate = 0.95

                    lrate = ExponentialDecay(
                        initial_lr,
                        decay_steps=steps,
                        decay_rate=decay_rate,
                        staircase=True)


                    model = build_model(args.model, lr=lrate, n_outputs=n,
                                        bneck=args.bottleneck,
                                        noise=cur_noise if cur_noise > 0 else None,
                                        act=args.act)

                    # lrate = LearningRateScheduler(step_decay, verbose=1)

                    # out_tracker = pick_output(x_train[:2], 29, y_train[:2])   

                    logs = model.fit(x_train, y_train,
                                     batch_size=args.batch_size,
                                     epochs=epochs,
                                     validation_data=(x_test, y_test),
                                     shuffle=True,
                                     verbose=2)#,
                            # callbacks = [lrate])#out_tracker])

                    su.export_hparams(logs, save_dir, 'params.json',
                                      {'num_classes': n, 'num_per_class': x,
                                       'limit_output': args.limit,
                                       'model': args.model, 'batch_norm': True,
                                       'data': 'rnd-gauss',
                                       'bottleneck': args.bottleneck,
                                       'noise': cur_noise,
                                       'standardized': args.standardize,
                                       'activation': args.act}
                                     )

                    model.save_weights(os.path.join(save_dir, 'model',
                                                    'model.h5'))
                    logs.model = None

                    pickle.dump(logs, open(os.path.join(save_dir, 'logs',
                                                        'logs.p'), 'wb'))
                    pickle.dump(rnd, open(os.path.join(save_dir, 'rnd.p'),
                                         'wb'))
                    plot_results(logs, 'rnd-gauss', n, args.model, save_dir)
                    
                    del model
                    gc.collect()
                    
                    cur_noise += 0.1 if args.act == 'relu' else 0.02
