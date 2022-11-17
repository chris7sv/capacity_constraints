import pickle
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
from tensorflow.keras.layers import Dense,Flatten, Input, \
     GlobalAveragePooling2D
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback, LearningRateScheduler, \
     TensorBoard, CSVLogger
from tensorflow.keras.utils import plot_model
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from ..src.image_utils import * #iu
from ..src.saving_util import * #su
from ..src.models import *
from ..src.model_utils import *

"""
Script related to results from section 3.1, using networks pre-trained on Imagenet

Code by Chris Tsvetkov
"""

def generate_random_image(num_samples, data_shape, rnd,
                          random_type='rnd-gauss', means=None, std=None):

    """Generates new random pixel image"""

    data = np.ndarray(
        [
            num_samples,
            data_shape[0],
            data_shape[1],
            data_shape[2]
        ],
        dtype='float32')

    if random_type == 'rnd-gauss':
        for i in range(num_samples):
            if means is None:
                data[i] = rnd.normal(loc=120.75, scale=69.93,
                                     size=(data_shape[0]
                                           *data_shape[1]
                                           *data_shape[2])
                                     ).reshape(
                                         data_shape[0],
                                         data_shape[1],
                                         data_shape[2])

            elif isinstance(means, (list, tuple)):
                data_point = np.ndarray(shape=data_shape)
                for idx, stat in enumerate(zip(means, std)):
                    data_point[:, :, idx] = rnd.normal(stat[0], stat[1],
                                                       (data_shape[0]*
                                                        data_shape[1])) \
                                                        .reshape(data_shape[0],
                                                                 data_shape[1])
                data[i] = np.copy(data_point)
    return data

def vectorize_labels(labels, vec_size, max_vec_size=10):

    """transform scalar labels to one-hot vectors of a given size"""

    new_labels = np.zeros([len(labels), vec_size], dtype='uint8')

    if vec_size < max_vec_size:        
        uniq = np.unique(labels)

        for idx, val in enumerate(uniq):
            labels[labels == val] = idx

    for i in range(len(labels)):
        new_labels[i, labels[i]] = 1

    return new_labels   

def create_labels(num_labels, vec_size):

    """Create <<num_labels>> aritficial labels from 0 to <<vec_size>>"""

    new_labels = np.zeros(shape=(num_labels, vec_size), dtype='uint8')

    for i in range(len(vec_size)):
        new_labels[i*int(num_labels/vec_size)+int(num_labels/vec_size),i] = 1

    np.random.shuffle(new_labels)

    return new_labels


def preprocess_x(x):
    """
    Preprocess input for ResNet or VGG. The mean and std are for CIFAR10

    Code adapted from keras.applications.image_utils
    """
    mean = [0.485, 0.456, 0.406]
    mean = [x*255 for x in mean]
    std = [0.229, 0.224, 0.225]
    std = [x*255 for x in std]
    x[..., 0] -= mean[0]
    x[..., 1] -= mean[1]
    x[..., 2] -= mean[2]
    # x[..., 0] /= std[0]
    # x[..., 1] /= std[1]
    # x[..., 2] /= std[2]
    # print(x.shape)
    # print(x[0,0,:])
    return x

# class CheckWeightNorms(Callback):

#     def __init__(self):
#         super().__init__()
#         self.pred = 
#         self.y_true
#         self.loss
#         self.grad
#     def call(self, epoch):
#         return

# class Gradient_norm(tf.keras.metrics.Metric):
#     """
#     Compute gradient norm. Impletmented as a custom metric below.
#     Code from: 
#     https://stackoverflow.com/questions/45694344/calculating-gradient-norm-wrt-weights-with-keras
#     """
#     def __init__(self, name='gradient_norm', **kwargs):
#         super(Gradient_norm, self).__init__(name=name, **kwargs)
#         self.grad_norm = self.add_weight(name='gn', initializer='zeros')

#     def update_state(self, y_pred, y_true, sample_weight=None):
#         self.
#     def result(self):

#     def reset_state(self):



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    help_ = "Choose model architecture. Default : inception"
    parser.add_argument('-m', '--model', help=help_, choices=['vgg', 'res', 'inc'])
    help_ = "Choose whether to load imagenet weights or train a  \
             random initialization"
    parser.add_argument('-w', '--weights', help=help_, type=str,
                        choices=['imagenet', None])
    help_ = "Choose whether to attach and train a new clasifier. \
             Defaluts to true"
    parser.add_argument('-t', '--top', help=help_, action='store_false')
    help_ = "Choose data for simulation."
    parser.add_argument('-d', '--data', type=str,
                        choices=['cifar', 'rnd-pixels', 'rnd-labels'])
    parser.add_argument('-r', '--repeat', type=int, default=1)
    help_ = "Choose whether to use original (large) size data or upscaled. \
             CIFAR data can only be upscaled"
    parser.add_argument('--data_size_type', type=str,
                        choices=['upscaled', 'large'], default='upscaled')
    parser.add_argument('-e', '--epochs', type=int, default=20)
    parser.add_argument('-f', '--freeze', action='store_true')
    help_ = "Freeze only (the top) half of the convolutional weights"
    parser.add_argument('--freeze_frac', help=help_, type=float, default=None)

    args = parser.parse_args() 

    set_session_params(tf.version.VERSION)
    K.clear_session()

    #TODO: Update all constant names to UPPER_CASE style

    rnd = np.random.RandomState()

    MODEL_NAME = args.model
    SIM = args.data
    DATA = 'random' if SIM == 'rnd-pixels' else 'cifar'
    DATA_SIZE_TYPE = args.data_size_type
    N_HIDDEN = 4096 if MODEL_NAME == 'vgg' else 512
    WEIGHTS = args.weights
    EPOCHS = args.epochs
    FREEZE_FRAC = args.freeze_frac
    FREEZE = '_freeze' if args.freeze else ''
    SIM_TYPE = '_fine_tune' if  WEIGHTS == 'imagenet' else '_no_pre'
    LR = 0.01 #0.0005

    exp_name = f'{MODEL_NAME}{FREEZE}_sim_{SIM}_{DATA_SIZE_TYPE}{SIM_TYPE}'


    for r in range(args.repeat):


        save_dir = store_experiment(os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir())), 'data'), exp_name)

        input_tensor = Input(shape=(224, 224, 3))
        #Initialize model
        if MODEL_NAME == 'vgg':
            base_model = VGG19(input_tensor=input_tensor, weights=WEIGHTS,
                               include_top=False)
        elif MODEL_NAME == 'res':
            base_model = ResNet50(input_tensor=input_tensor, weights=WEIGHTS,
                                  include_top=False)
        elif MODEL_NAME == 'inc':
            base_model = InceptionV3(input_tensor=input_tensor, weights=WEIGHTS,
                                     include_top=False)

        if args.freeze:
            for layer in base_model.layers:
                layer.trainable = False
            if FREEZE_FRAC:
                for layer in base_model.layers[
                        int(len(base_model.layers)*FREEZE_FRAC):]:
                    layer.trainable = True

        for layer in base_model.layers:
            if  len(layer.get_weights()) > 0:
                print(layer.name, '\t', 'weight norm:',
                      np.linalg.norm(layer.get_weights()[0]))
            if layer.trainable_weights:
                print('\t', 'trainable weights:',
                      tf.size(layer.trainable_weights[0]))

        x = base_model.output
        if MODEL_NAME == 'vgg':
            x = Flatten()(x)
            x = Dense(N_HIDDEN, activation='relu')(x)
            x = Dense(N_HIDDEN, activation='relu')(x)
        else:
            x = GlobalAveragePooling2D()(x)
            x = Flatten()(x)
            x = Dense(N_HIDDEN, activation='relu')(x)
            # x = Dense(N_HIDDEN, activation='relu')(x)
        out = Dense(10, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=out)
        model.summary()

        #TODO: Try adding momentum if networks aren't training.
        sgd = SGD(lr=LR, decay=1e-6)
        #0.0005 -vgg transfer #0.1 - resnet from scratch PREV: lr=0.00075
        model.compile(loss='categorical_crossentropy', optimizer=sgd,
                      metrics=['accuracy'])
   
        # x_train = generate_random_image(10000,(224,224,3),
        #                                 rnd=rnd,means=mean, std=std)

        #Load data, define data generators with preprocessing
        (_, y_train), (_, y_test) = cifar10.load_data()

        y_train = y_train.tolist()
        y_test = y_test.tolist()
        
        
        train_dir = os.path.join(os.getcwd(), f'{DATA}_{DATA_SIZE_TYPE}/train')
        img_paths_train = [os.path.join(train_dir, f'img_{k}.png')
                           for k in range(50000)]
        if SIM == 'rnd-labels':
            np.random.shuffle(y_train)
        df_dict_train = {'img_paths':img_paths_train, 'labels':y_train}

        test_dir = os.path.join(os.getcwd(), f'{DATA}_{DATA_SIZE_TYPE}/test')
        img_paths_test = [os.path.join(test_dir, f'img_{k}.png') 
                          for k in range(10000)]
        if SIM == 'rnd-labels':
            np.random.shuffle(y_test)
        df_dict_test = {'img_paths':img_paths_test, 'labels':y_test}

        # print(y_train[:10])

        df_train = pd.DataFrame(df_dict_train)
        df_test = pd.DataFrame(df_dict_test)

        datagen = ImageDataGenerator(preprocessing_function=preprocess_x)
        train_generator = datagen.flow_from_dataframe(df_train,
                                                      x_col="img_paths",
                                                      y_col="labels",
                                                      batch_size=25,
                                                      target_size=(224, 224))
                                                      #, validate_filenames=False)
        # test_generator = datagen.flow_from_dataframe(df_test,
        #                                              x_col="img_paths",
        #                                              y_col="labels",
        #                                              batch_size=32,
        #                                              target_size=(224,224))
        #                                              #, validate_filenames=False)

        STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
        # STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

        tensorboard_cb = TensorBoard(log_dir=save_dir)
        csv_logger = CSVLogger(os.path.join(save_dir, 'logs', 'training.log'))

        #Train model

        logs = model.fit(train_generator, steps_per_epoch=STEP_SIZE_TRAIN,
                         epochs=EPOCHS, verbose=2,
                         callbacks=[tensorboard_cb, csv_logger])
                         # validation_data=test_generator,
                         #  validation_steps=STEP_SIZE_TEST,

        #Save weights, logs, hyperparameters and simulation details

        model.save_weights(os.path.join(save_dir, 'model', 'model.h5'))

        export_hparams(logs, save_dir, 'params.json',
                          {'data':SIM, 'model':MODEL_NAME, 'weights':WEIGHTS,
                          'freeze':args.freeze, 'freeze_fraction':FREEZE_FRAC,
                          'data_sizing_type':DATA_SIZE_TYPE,
                          'hidden_1':N_HIDDEN,
                          'hidden_2':N_HIDDEN if MODEL_NAME == 'vgg' else None})
        logs.model = None
        
        pickle.dump(logs, open(os.path.join(save_dir, 'logs', 'logs.p'), 'wb'))

        K.clear_session()