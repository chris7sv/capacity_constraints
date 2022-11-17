# -*- coding: utf-8 -*-
"""
Created on Thu May  2 23:29:50 2019

@author: Chris Tsvetkov
"""
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Layer, Input, Conv2D, BatchNormalization, \
    MaxPooling2D, Dense, Flatten, Dropout, Activation, GlobalAveragePooling2D, \
    concatenate, AveragePooling2D, LeakyReLU, GaussianNoise, Cropping2D, Lambda, \
    Concatenate, ZeroPadding2D, DepthwiseConv2D, Reshape
import tensorflow.keras.applications as ka
from tensorflow.keras.regularizers import l2

from tensorflow.keras.models import Model, Sequential

import tensorflow.keras.backend as K

# from tensorflow.keras.layers import VersionAwareLayers
# from tensorflow.keras.utils import layer_utils


#from keras.activations import softmax, relu
#from keras import regularizers


class LimitPrecision(Layer):
    """
    Round the output of layer activation to reduce precision to specified
    Number of decimals.

    DOES NOT WORK!!!
    """
    def __init__(self, dp, **kwargs):
        super(LimitPrecision, self).__init__(**kwargs)
        self._dp = dp
        
    def build(self, input_shape):       
        self.shape = input_shape
        super(LimitPrecision, self).build(input_shape)
        
    def call(self, inputs):
        x = inputs
        precision = self._dp
        multiplier = K.constant(10**precision,dtype='float16')
#        result = tf.cond((tf.equal(tf.math.abs(x),
#                         tf.constant(0, dtype=tf.float16))),
#                         lambda: x, lambda: (K.round(x*multiplier))/multiplier)
                         
#        if tf.math.is_inf(x) or tf.math.is_nan(x):
#           return x
#        print(K.dtype(K.round(x*multiplier)/multiplier))
#        #return (K.round(x*multiplier))/multiplier 
#        #return round(x, precision-int(np.floor(np.log10(abs(x))))-1)
        return (K.round(x*multiplier))/multiplier


# def LimitFP_v2(x,prec):


    
def log10(x):
  numerator = tf.math.log(x)
  denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator

def round_through(x):
   '''
   Element-wise rounding to the closest integer with full 
   gradient propagation.
   A trick from [Sergey Ioffe](http://stackoverflow.com/a/36480182)
   '''
   rounded = K.round(x)
   return x + K.stop_gradient(rounded - x) 

# TODO: Just delete function below, completely incomprehensible after edits
   
def LimFP(x, prec):
    mult = 10**prec
#    print(K.dtype(K.round(var[0]*mult)))
#    result = tf.cond(tf.math.is_nan(var[0]),
#                     lambda:var[0], lambda: (K.round(var[0]*mult)/mult))
#    rounded = K.round(x*mult)/mult
#    x = ((x*mult)//K.constant(1.0))/mult
#    scaled = x*mult
#    y = K.placeholder(shape=x.shape)
#    rounding = K.function([y],[K.round(y*(10**prec))])
#    rounded = rounding([x])
#    resid = scaled % K.constant(1)
#    rounded = (scaled-resid)/mult
#    final = x - resid + K.constant(trunc)
#(scaled - (scaled % K.constant(1)))/mult#rounded#((x*mult)//K.constant(1.0))/mult#
# x + K.stop_gradient(rounded - x)
#  #round(var[0], var[1]-tf.math.round(tf.math.floor(log10(tf.abs(var[0]))))-1)
    return K.in_test_phase(K.round(x*mult)/mult, x)#

class LRN2D(Layer):
    """
    This code is adapted from pylearn2.
    License at: https://github.com/lisa-lab/pylearn2/blob/master/LICENSE.txt
    """

    def __init__(self, alpha=1e-4, k=2, beta=0.75, n=5, **kwargs):
        if n % 2 == 0:
            raise NotImplementedError("LRN2D only works with odd n. n provided: " + str(n))
        super(LRN2D, self).__init__(**kwargs)
        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.n = n

    def get_output(self, train):
        X = self.get_input(train)
        b, r, c, ch = K.shape(X)
        half_n = self.n // 2
        input_sqr = K.square(X)
        extra_channels = K.zeros((b, r, c,  ch + 2*half_n))
        input_sqr = K.concatenate([extra_channels[:, :half_n, :, :],
                                   input_sqr,
                                   extra_channels[:, half_n + ch:, :, :]],
                                  axis=3)
        scale = self.k
        for i in range(self.n):
            scale += self.alpha * input_sqr[:, i:i + ch, :, :]
        scale = scale**self.beta
        return X/scale

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "alpha": self.alpha,
                  "k": self.k,
                  "beta": self.beta,
                  "n": self.n}
        base_config = super(LRN2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class LocalResponseNormalization(Layer):
    
    def __init__(self, n=5, alpha = 5e-4, beta = 0.75, k=2, **kwargs):
        
        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.k = k
        super(LocalResponseNormalization, self).__init__(**kwargs)
    
    def build (self, input_shape):
        
        self.shape = input_shape
        super(LocalResponseNormalization, self).build(input_shape)
        
    def call(self, x, mask=None):
        if K.image_data_format == "channels_first":
            
            _, f,r,c = self.shape
            
        else:
            _, r,c,f = self.shape

            
        squared = K.square(x)
        
        pooled = K.pool2d(squared, (self.n, self.n), strides = (1,1),
                          padding = "same", pool_mode = "avg")
        

        if K.image_data_format == "channels_first":
            summed = K.sum(pooled, axis=1, keepdims=True)
            averaged = self.alpha*K.repeat_elements(summed, f, axis=1)
        else:
            summed = K.sum(pooled, axis=3, keepdims=True)

            averaged = self.alpha*K.repeat_elements(summed, f, axis=3)
            

        denom = K.pow(self.k + averaged, self.beta)

        
        return x / denom
    
    def get_output_shape_for(self, input_shape):
        
        return input_shape

def ConvModel():  

    """
    A standard convolutional neural network used in very early pilot sutides.
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=(32,32,3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    return model

def VGG16():
    vgg = ka.VGG16(include_top=False, weights=None, input_shape=(32, 32, 3))
    
    out = vgg.output
    out = GlobalAveragePooling2D()(out)
    top = Dense(1024, activation='relu')(out)
    top = Dense(10, activation='softmax')(top)
    
    model = Model(vgg.input, output=top)
    
    return model

def conv_module(filters, kernel, stride, prev_layer, bn, noise, prec=None,
        act='relu', wd=0.0):
    
    layer = Conv2D(filters, kernel, strides=(stride, stride), padding='same',
                   kernel_initializer='glorot_uniform',
                   kernel_regularizer=l2(wd),
                   bias_regularizer=l2(wd))(prev_layer)
    if prec is not None:
        layer = Lambda(LimFP, arguments={'prec':prec},
                        trainable=False)((layer))  
        #layer = LimitPrecision(prec)(layer)

    if bn:
        layer = BatchNormalization()(layer)    
    layer = Activation(act)(layer)
    if prec is not None:
        layer = layer = Lambda(LimFP, arguments={'prec':prec},
                                trainable=False)((layer))
         #layer = LimitPrecision(prec)(layer)     
    if noise is not None:
        #Set training to true for normal operation
        layer = GaussianNoise(noise)(layer, training=True)
        if prec is not None:
            layer = Lambda(LimFP, arguments={'prec':prec},
                            trainable=False)((layer))
            #layer = LimitPrecision(prec)(layer)
 
    return layer

def inception_module(ch1_filters, ch2_filters, prev_layer, bn, noise,
         prec=None, act='relu'):
    conv1 = conv_module(ch1_filters, 1, 1, prev_layer, bn, noise, prec, act)
    conv2 = conv_module(ch2_filters, 3, 1, prev_layer, bn, noise, prec, act)
    merge = concatenate([conv1, conv2])
    if prec is not None:
        limit = Lambda(LimFP, arguments={'prec':prec},
                        trainable=False)((merge))
        #limit = LimitPrecision(prec)(merge)
    return merge if prec is None else limit

def downsample_module(filters, prev_layer, bn, noise, prec, act):
    conv = conv_module(filters, 3, 2, prev_layer, bn, noise, prec, act)
    pool = MaxPooling2D((3, 3), (2, 2), padding='same')(prev_layer)
    if prec is not None:
        pool = Lambda(LimFP, arguments={'prec':prec},
                      trainable=False)((pool))
        #LimitPrecision(prec)(pool)
    merge = concatenate([conv, pool])
    if prec is not None: 
        limit = Lambda(LimFP, arguments={'prec':prec},
                       trainable=False)((merge))
        #LimitPrecision(prec)(merge)
    return merge if prec is None else limit

def inception_small(bn, n_outputs=10, bneck=False, bneck_width=2, noise=None, drop=False,
        prec=None, act='relu', wd=0.0):
    in_layer = Input(shape=(32, 32, 3))
#    crop = Cropping2D(2)(in_layer)
    if not bneck:
        conv = conv_module(96, 3, 1, in_layer, bn, noise, prec, act)
    else:
        conv = conv_module(bneck_width, 3, 1, in_layer, bn, noise, prec, act)
    
    inc1 = inception_module(32, 32, conv, bn, noise, prec, act) #if not bneck else inception_module(2,2, conv, bn, noise)         
    inc2 = inception_module(32, 48, inc1, bn, noise, prec, act)

    down1 = downsample_module(80, inc2, bn, noise, prec, act)
   
    inc3 = inception_module(112, 48, down1, bn, noise, prec, act)
    inc4 = inception_module(96, 64, inc3, bn, noise, prec, act)
    inc5 = inception_module(80, 80, inc4, bn, noise, prec, act)
    inc6 = inception_module(48, 96, inc5, bn, noise, prec, act)
    down2 = downsample_module(96, inc6, bn, noise, prec, act)
    
    inc7 = inception_module(176, 160, down2, bn, noise, prec, act)
    inc8 = inception_module(176, 160, inc7, bn, noise, prec, act)
    pool = AveragePooling2D((7, 7), 1)(inc8)
    if prec is not None:
        pool = Lambda(LimFP, arguments={'prec':prec},
                    trainable=False)((pool))
    pool  = Flatten()(pool)
    out_layer = Dense(n_outputs, activation='softmax')(pool)
    
    model = Model(in_layer, out_layer)
    
    return model

def vgg_net(n_outputs=10, bneck=False, noise=None):
    bn = False
    in_layer = Input(shape=(32, 32, 3))
    conv1 = conv_module(64, 3, 1, in_layer, bn, noise)
    if bneck:
        conv2 = conv_module(3, 3, 1, conv1, bn, noise)
    else:   
        conv2 = conv_module(64, 3, 1, conv1, bn, noise)
    maxpool = MaxPooling2D((2, 2), 2)(conv2)
    

    conv3 = conv_module(128, 3, 1,maxpool, bn, noise)
    conv4 = conv_module(128, 3, 1, conv3, bn, noise)
    maxpool2 = MaxPooling2D((2, 2), 2)(conv4)
    
    conv5 = conv_module(256, 3, 1, maxpool2, bn, noise)
    conv6 = conv_module(256, 3, 1, conv5, bn, noise)
    maxpool3 = MaxPooling2D((2, 2), 2)(conv6)
    
    conv7 = conv_module(512, 3, 1, maxpool3, bn, noise)
    conv8 = conv_module(512, 3, 1, conv7, bn, noise)
    maxpool4 = MaxPooling2D((2, 2), 2)(conv8)
    
    conv9 = conv_module(512, 3, 1,maxpool4, bn, noise)
    conv10 = conv_module(512, 3, 1,conv9, bn, noise)
    maxpool5 = MaxPooling2D((2, 2), 2)(conv10)

    
#    glob = GlobalAveragePooling2D()(maxpool5)
    
    flattened = Flatten()(maxpool5)
    
    dense1 = Dense(4096, activation='relu')(flattened)
#    drop1 = Dropout(0.2)(dense1)
    dense2 = Dense(4096, activation='relu')(dense1)
#    drop2 = Dropout(0.2)(dense2)
    out_layer = Dense(n_outputs, activation='softmax')(dense2)
    
    
    model = Model(in_layer, out_layer)
    
    return model
    

def alexnet_small(n_outputs=10, bneck=False, bneck_width=8, noise=None, drop=False,
        prec=None, act='relu', wd=0.0):
    
    in_layer = Input(shape=(32, 32, 3))
    crop = Cropping2D(2)(in_layer)
    if not bneck:
        conv1 = conv_module(200, 5, 1, crop, bn=True, noise=noise, prec=prec,
                            act=act, wd=wd) 
    else:
         conv1 = conv_module(bneck_width, 5, 1, crop, bn=True, noise=noise, prec=prec,
                             act=act, wd=wd)
    pool1 = MaxPooling2D((3,3))(conv1)
    #lrn1 = LRN2D()(pool1)
    if drop: drop1 = Dropout(0.6)(pool1)
    #if bneck:
        # conv2 = conv_module(16,5,1,lrn1, bn=True, noise=noise, prec=prec, act=act)    
    # else:
    conv2 = conv_module(200, 5, 1, drop1 if drop else pool1, bn=True,
                        noise=noise, prec=prec, act=act,wd=wd)    
    pool2 = MaxPooling2D((3, 3), name='last')(conv2)
    #lrn2 = LRN2D()(pool2)

    flat = Flatten()(pool2)
    if drop: drop2 = Dropout(0.6)(flat)
    # if prec is not None: lp1 = LimitPrecision(prec)(drop2)
    dense1 = Dense(384, kernel_regularizer=l2(wd),
                   bias_regularizer=l2(wd))(drop2 if drop else flat)
    bn1 = BatchNormalization()(dense1)

    # if prec is not None: lp2 = LimitPrecision(prec)(dense1)
    relu3 = Activation(act)(bn1)# if prec is None else lp2)
    
    # if prec is not None: lp3 = LimitPrecision(prec)(relu3)
    if drop: drop3 = Dropout(0.6)(relu3)
    if noise is not None: noise1 = GaussianNoise(noise)(relu3)
    dense2 = Dense(192, kernel_regularizer=l2(wd),
                   bias_regularizer=l2(wd))(drop3 if drop else \
                                            noise1 if noise is not None \
                                            else relu3)
    # bn2 = BatchNormalization()(dense2)
    # if noise is not None: noise2 = GaussianNoise(noise)(dense2)
    bn2 = BatchNormalization()(dense2)

    # if prec is not None: lp4 = LimitPrecision(prec)(dense2)
    relu4 = Activation(act)(bn2)# if prec is None else lp4)
    if drop: drop4 = Dropout(0.6)(relu4) 
    if noise is not None: noise2 = GaussianNoise(noise)(relu4)
    # if prec is not None: lp5 = LimitPrecision(prec)(relu4)
    dense3 = Dense(n_outputs)(drop4 if drop else noise2 if noise is not None \
                              else relu4)
    # if prec is not None: lp6 = LimitPrecision(prec)(dense3)
    softmax = Activation('softmax')(dense3)# if prec is None else lp6)

    model = Model(in_layer, softmax)   

    return model

#def alt_inception_small(bn, n_outputs, bneck=False, noise=None, prec=None, act='relu'):
#    in_layer = Input(shape=(32,32,3))
##    crop = Cropping2D(2)(in_layer)
#    conv = conv_module(96,3,1,in_layer, bn, noise, prec, act) if not bneck else conv_module(2,3,1,in_layer, bn, noise, prec, act)
#    
#    cm1 = Model(in_layer, conv) 
#    input_1 = np.round(cm1.predict(in_layer), 2)
#    in_2 = Input(shape=(conv.shape))
#    inc1 = inception_module(32,32, conv, bn, noise, prec, act) #if not bneck else inception_module(2,2, conv, bn, noise)         
#    inc2 = inception_module(32,48, inc1, bn, noise, prec, act) #if not bneck else inception_module(2,2, inc1, bn, noise) 
#    if bneck:
#        noise = None
#    down1 = downsample_module(80, inc2, bn, noise, prec, act)
#   
#    inc3 = inception_module(112,48, down1, bn, noise, prec, act)
#    inc4 = inception_module(96,64, inc3, bn, noise, prec, act)
#    inc5 = inception_module(80,80, inc4, bn, noise, prec, act)
#    inc6 = inception_module(48, 96, inc5, bn, noise, prec, act)
#    down2 = downsample_module(96, inc6, bn, noise, prec, act)
#    
#    inc7 = inception_module(176,160, down2, bn, noise, prec, act)
#    inc8 = inception_module(176,160, inc7, bn, noise, prec, act)
#    pool = AveragePooling2D((7,7),1)(inc8)
#    if prec is not None: pool=Lambda(LimFP, arguments={'prec':prec}, trainable=False)((pool))#LimitPrecision(prec)(pool)
#    pool  = Flatten()(pool)
#    out_layer = Dense(n_outputs, activation='softmax')(pool)
#    
#    model = Model(in_layer, out_layer)
#    
#    return model    

"""
DenseNet code copied and adapted from Keras appications 
URL: https://github.com/keras-team/keras/blob/v2.9.0/keras/applications/densenet.py
"""

# layers = VersionAwareLayers()

def dense_block(x, blocks, name, noise=0.0, activation='relu', bottleneck=False):
    """A dense block.
    Args:
    x: input tensor.
    blocks: integer, the number of building blocks.
    name: string, block label.
    Returns:
    Output tensor for the block.
    """
    for i in range(blocks):
        if i==0:
            x = conv_block(x, 32, name=name + '_block' + str(i + 1),
                        noise=noise, activation=activation, bottleneck=bottleneck)
        else:
            x = conv_block(x, 32, name=name + '_block' + str(i + 1),
                        noise=noise, activation=activation)
    return x

def transition_block(x, reduction, name, noise=0.0, activation='relu'):
    """A transition block.
    Args:
    x: input tensor.
    reduction: float, compression rate at transition layers.
    name: string, block label.
    Returns:
    output tensor for the block.
    """
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
    x = BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_bn')(x)
    x = Activation(activation, name=name + f'_{activation}')(x)
    if noise:
        x = GaussianNoise(noise)(x, training=True)
    x = Conv2D(int(K.int_shape(x)[bn_axis] * reduction),
                    1,
                    use_bias=False,
                    name=name + '_conv')(x)
    x = AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    return x


def conv_block(x, growth_rate, name, noise=0.0, activation='relu', bottleneck=False):
    """A building block for a dense block.
    Args:
    x: input tensor.
    growth_rate: float, growth rate at dense layers.
    name: string, block label.
    Returns:
    Output tensor for the block.
    """
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
    x1 = BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(
            x)
    x1 = Activation(activation, name=name + f'_0_{activation}')(x1)
    if noise:
        x1 = GaussianNoise(noise)(x1, training=True)
    if bottleneck:      # Hard code number of channels in bottleneck layer
         x1 = Conv2D(
                4 * 1 , 1, use_bias=False, name=name + '_1_conv')(
                x1)
    else:
        x1 = Conv2D(
                4 * growth_rate, 1, use_bias=False, name=name + '_1_conv')(
                x1)
    x1 = BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(
            x1)
    x1 = Activation(activation, name=name + f'_1_{activation}')(x1)
    if noise:
        x1 = GaussianNoise(noise)(x1, training=True)
    x1 = Conv2D(
        growth_rate, 3, padding='same', use_bias=False, name=name + '_2_conv')(
            x1)
    x = Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x


def DenseNet(
    blocks,
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=10,
    classifier_activation='softmax',
    noise=0.0,
    activation='relu',
    bottleneck=False,
    ):
    """Instantiates the DenseNet architecture.
    Reference:
    - [Densely Connected Convolutional Networks](
    https://arxiv.org/abs/1608.06993) (CVPR 2017)
    This function returns a Keras image classification model,
    optionally loaded with weights pre-trained on ImageNet.
    For image classification use cases, see
    [this page for detailed examples](
    https://keras.io/api/applications/#usage-examples-for-image-classification-models).
    For transfer learning use cases, make sure to read the
    [guide to transfer learning & fine-tuning](
    https://keras.io/guides/transfer_learning/).
    Note: each Keras Application expects a specific kind of input preprocessing.
    For DenseNet, call `tf.keras.applications.densenet.preprocess_input` on your
    inputs before passing them to the model.
    `densenet.preprocess_input` will scale pixels between 0 and 1 and then
    will normalize each channel with respect to the ImageNet dataset statistics.
    Args:
    blocks: numbers of building blocks for the four dense layers.
    include_top: whether to include the fully-connected
    layer at the top of the network.
    weights: one of `None` (random initialization),
    'imagenet' (pre-training on ImageNet),
    or the path to the weights file to be loaded.
    input_tensor: optional Keras tensor
    (i.e. output of `layers.Input()`)
    to use as image input for the model.
    input_shape: optional shape tuple, only to be specified
    if `include_top` is False (otherwise the input shape
    has to be `(224, 224, 3)` (with `'channels_last'` data format)
    or `(3, 224, 224)` (with `'channels_first'` data format).
    It should have exactly 3 inputs channels,
    and width and height should be no smaller than 32.
    E.g. `(200, 200, 3)` would be one valid value.
    pooling: optional pooling mode for feature extraction
    when `include_top` is `False`.
    - `None` means that the output of the model will be
        the 4D tensor output of the
        last convolutional block.
    - `avg` means that global average pooling
        will be applied to the output of the
        last convolutional block, and thus
        the output of the model will be a 2D tensor.
    - `max` means that global max pooling will
        be applied.
    classes: optional number of classes to classify images
    into, only to be specified if `include_top` is True, and
    if no `weights` argument is specified.
    classifier_activation: A `str` or callable. The activation function to use
    on the "top" layer. Ignored unless `include_top=True`. Set
    `classifier_activation=None` to return the logits of the "top" layer.
    When loading pretrained weights, `classifier_activation` can only
    be `None` or `"softmax"`.
    Returns:
    A `keras.Model` instance.
    """

    # Determine proper input shape
    input_shape = input_shape 
    #ka.imagenet_utils.obtain_input_shape(input_shape,
                                                    # default_size=224,
                                                    # min_size=32,
                                                    # data_format=K.image_data_format(),
                                                    # require_flatten=False,
                                                    # weights=None)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1

    x = ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
    # if bottleneck:
        # x = Conv2D(2, 7, strides=2, use_bias=False, name='conv1/conv')(x)
    # else:
    x = Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='conv1/bn')(x)
    x = Activation(activation, name=f'conv1/{activation}')(x)
    if noise:
        x = GaussianNoise(noise)(x, training=True)
    x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = MaxPooling2D(3, strides=2, name='pool1')(x)

    x = dense_block(x, blocks[0], name='conv2', noise=noise, activation=activation,
                    bottleneck=bottleneck)
    x = transition_block(x, 0.5, name='pool2', noise=noise, activation=activation)
    x = dense_block(x, blocks[1], name='conv3', noise=noise, activation=activation)
    x = transition_block(x, 0.5, name='pool3', noise=noise, activation=activation)
    x = dense_block(x, blocks[2], name='conv4', noise=noise, activation=activation)
    x = transition_block(x, 0.5, name='pool4', noise=noise, activation=activation)
    x = dense_block(x, blocks[3], name='conv5', noise=noise, activation=activation)

    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='bn')(x)
    x = Activation(activation, name=f'{activation}')(x)
    if noise:
        x = GaussianNoise(noise)(x, training=True)

    x = GlobalAveragePooling2D(name='avg_pool')(x)

    # ka.imagenet_utils.validate_activation(classifier_activation, weights)
    x = Dense(classes, activation=classifier_activation,
                name='predictions')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    # if input_tensor is not None:
    #     inputs = layer_utils.get_source_inputs(input_tensor)
    # else:
    inputs = img_input

    # Create model.
    if blocks == [6, 12, 24, 16]:
        model = Model(inputs, x, name='densenet121')
    elif blocks == [6, 12, 32, 32]:
        model = Model(inputs, x, name='densenet169')
    elif blocks == [6, 12, 48, 32]:
        model = Model(inputs, x, name='densenet201')
    else:
        model = Model(inputs, x, name='densenet')

    return model

def MobileNet(input_shape=None,
              alpha=1.0,
              depth_multiplier=1,
              dropout=1e-3,

              input_tensor=None,
              pooling=None,
              classes=10,
              classifier_activation='softmax',
              activation='relu',
              noise=0.0,
              bottleneck=False,
              **kwargs):
    """Instantiates the MobileNet architecture.
    Reference:
    - [MobileNets: Efficient Convolutional Neural Networks
        for Mobile Vision Applications](
        https://arxiv.org/abs/1704.04861)
    This function returns a Keras image classification model,
    optionally loaded with weights pre-trained on ImageNet.
    For image classification use cases, see
    [this page for detailed examples](
        https://keras.io/api/applications/#usage-examples-for-image-classification-models).
    For transfer learning use cases, make sure to read the
    [guide to transfer learning & fine-tuning](
        https://keras.io/guides/transfer_learning/).
    Note: each Keras Application expects a specific kind of input preprocessing.
    For MobileNet, call `tf.keras.applications.mobilenet.preprocess_input`
    on your inputs before passing them to the model.
    `mobilenet.preprocess_input` will scale input pixels between -1 and 1.
    Args:
        input_shape: Optional shape tuple, only to be specified if `include_top`
        is False (otherwise the input shape has to be `(224, 224, 3)` (with
        `channels_last` data format) or (3, 224, 224) (with `channels_first`
        data format). It should have exactly 3 inputs channels, and width and
        height should be no smaller than 32. E.g. `(200, 200, 3)` would be one
        valid value. Default to `None`.
        `input_shape` will be ignored if the `input_tensor` is provided.
        alpha: Controls the width of the network. This is known as the width
        multiplier in the MobileNet paper. - If `alpha` < 1.0, proportionally
        decreases the number of filters in each layer. - If `alpha` > 1.0,
        proportionally increases the number of filters in each layer. - If
        `alpha` = 1, default number of filters from the paper are used at each
        layer. Default to 1.0.
        depth_multiplier: Depth multiplier for depthwise convolution. This is
        called the resolution multiplier in the MobileNet paper. Default to 1.0.
        dropout: Dropout rate. Default to 0.001.
        include_top: Boolean, whether to include the fully-connected layer at the
        top of the network. Default to `True`.
        weights: One of `None` (random initialization), 'imagenet' (pre-training
        on ImageNet), or the path to the weights file to be loaded. Default to
        `imagenet`.
        input_tensor: Optional Keras tensor (i.e. output of `layers.Input()`) to
        use as image input for the model. `input_tensor` is useful for sharing
        inputs between multiple different networks. Default to None.
        pooling: Optional pooling mode for feature extraction when `include_top`
        is `False`.
        - `None` (default) means that the output of the model will be
            the 4D tensor output of the last convolutional block.
        - `avg` means that global average pooling
            will be applied to the output of the
            last convolutional block, and thus
            the output of the model will be a 2D tensor.
        - `max` means that global max pooling will be applied.
        classes: Optional number of classes to classify images into, only to be
        specified if `include_top` is True, and if no `weights` argument is
        specified. Defaults to 1000.
        classifier_activation: A `str` or callable. The activation function to use
        on the "top" layer. Ignored unless `include_top=True`. Set
        `classifier_activation=None` to return the logits of the "top" layer.
        When loading pretrained weights, `classifier_activation` can only
        be `None` or `"softmax"`.
        **kwargs: For backwards compatibility only.
    Returns:
        A `keras.Model` instance.
    """
    
    # Determine proper input shape and default size.
    if input_shape is None:
        default_size = 224
    else:
        if K.image_data_format() == 'channels_first':
            rows = input_shape[1]
            cols = input_shape[2]
        else:
            rows = input_shape[0]
            cols = input_shape[1]

        if rows == cols and rows in [128, 160, 192, 224]:
            default_size = rows
        else:
            default_size = 224

    if K.image_data_format() == 'channels_last':
        row_axis, col_axis = (0, 1)
    else:
        row_axis, col_axis = (1, 2)
    rows = input_shape[row_axis]
    cols = input_shape[col_axis]

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = _conv_block(img_input, 32, alpha, strides=(2, 2),
                    activation=activation, noise=noise)
    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1,
                              activation=activation, noise=noise, bottleneck=bottleneck)

    x = _depthwise_conv_block(
        x, 128, alpha, depth_multiplier, strides=(2, 2), block_id=2,
        activation=activation, noise=noise)
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3,
                              activation=activation, noise=noise)

    x = _depthwise_conv_block(
        x, 256, alpha, depth_multiplier, strides=(2, 2), block_id=4,
        activation=activation, noise=noise)
    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5,
                              activation=activation, noise=noise)

    x = _depthwise_conv_block(
        x, 512, alpha, depth_multiplier, strides=(2, 2), block_id=6,
        activation=activation, noise=noise)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=7,
                              activation=activation, noise=noise)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=8,
                              activation=activation, noise=noise)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=9,
                              activation=activation, noise=noise)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=10,
                              activation=activation, noise=noise)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=11,
                              activation=activation, noise=noise)

    x = _depthwise_conv_block(
        x, 1024, alpha, depth_multiplier, strides=(2, 2), block_id=12,
        activation=activation, noise=noise)
    x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier, block_id=13,
                              activation=activation, noise=noise)

    x = GlobalAveragePooling2D(keepdims=True)(x)
    # x = Dropout(dropout, name='dropout')(x)
    x = Conv2D(classes, (1, 1), padding='same', name='conv_preds')(x)
    x = Reshape((classes,), name='reshape_2')(x)
    # imagenet_utils.validate_activation(classifier_activation, weights)
    x = Activation(activation=classifier_activation,
                            name='predictions')(x)


    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    # if input_tensor is not None:
    #     inputs = layer_utils.get_source_inputs(input_tensor)
    # else:
    inputs = img_input

    # Create model.
    model = Model(inputs, x, name='mobilenet_%0.2f_%s' % (alpha, rows))

    return model


def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1), noise=0.0, activation='relu'):
  """Adds an initial convolution layer (with batch normalization and relu6).
  Args:
    inputs: Input tensor of shape `(rows, cols, 3)` (with `channels_last`
      data format) or (3, rows, cols) (with `channels_first` data format).
      It should have exactly 3 inputs channels, and width and height should
      be no smaller than 32. E.g. `(224, 224, 3)` would be one valid value.
    filters: Integer, the dimensionality of the output space (i.e. the
      number of output filters in the convolution).
    alpha: controls the width of the network. - If `alpha` < 1.0,
      proportionally decreases the number of filters in each layer. - If
      `alpha` > 1.0, proportionally increases the number of filters in each
      layer. - If `alpha` = 1, default number of filters from the paper are
      used at each layer.
    kernel: An integer or tuple/list of 2 integers, specifying the width and
      height of the 2D convolution window. Can be a single integer to
      specify the same value for all spatial dimensions.
    strides: An integer or tuple/list of 2 integers, specifying the strides
      of the convolution along the width and height. Can be a single integer
      to specify the same value for all spatial dimensions. Specifying any
      stride value != 1 is incompatible with specifying any `dilation_rate`
      value != 1. # Input shape
    4D tensor with shape: `(samples, channels, rows, cols)` if
      data_format='channels_first'
    or 4D tensor with shape: `(samples, rows, cols, channels)` if
      data_format='channels_last'. # Output shape
    4D tensor with shape: `(samples, filters, new_rows, new_cols)` if
      data_format='channels_first'
    or 4D tensor with shape: `(samples, new_rows, new_cols, filters)` if
      data_format='channels_last'. `rows` and `cols` values might have
      changed due to stride.
  Returns:
    Output tensor of block.
  """
  channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
  filters = int(filters * alpha)
  x = Conv2D(
      filters,
      kernel,
      padding='same',
      use_bias=False,
      strides=strides,
      name='conv1')(inputs)
  x = BatchNormalization(axis=channel_axis, name='conv1_bn')(x)
  x = Activation(activation, name=f'conv1_{activation}')(x) #layers.ReLU(6., name='conv1_relu')(x)
  if noise:
    x = GaussianNoise(noise)(x)

  return x

def _depthwise_conv_block(inputs,
                          pointwise_conv_filters,
                          alpha,
                          depth_multiplier=1,
                          strides=(1, 1),
                          block_id=1,
                          activation='relu',
                          bottleneck=False,
                          noise=0.0):
    """Adds a depthwise convolution block.
    A depthwise convolution block consists of a depthwise conv,
    batch normalization, relu6, pointwise convolution,
    batch normalization and relu6 activation.
    Args:
        inputs: Input tensor of shape `(rows, cols, channels)` (with
        `channels_last` data format) or (channels, rows, cols) (with
        `channels_first` data format).
        pointwise_conv_filters: Integer, the dimensionality of the output space
        (i.e. the number of output filters in the pointwise convolution).
        alpha: controls the width of the network. - If `alpha` < 1.0,
        proportionally decreases the number of filters in each layer. - If
        `alpha` > 1.0, proportionally increases the number of filters in each
        layer. - If `alpha` = 1, default number of filters from the paper are
        used at each layer.
        depth_multiplier: The number of depthwise convolution output channels
        for each input channel. The total number of depthwise convolution
        output channels will be equal to `filters_in * depth_multiplier`.
        strides: An integer or tuple/list of 2 integers, specifying the strides
        of the convolution along the width and height. Can be a single integer
        to specify the same value for all spatial dimensions. Specifying any
        stride value != 1 is incompatible with specifying any `dilation_rate`
        value != 1.
        block_id: Integer, a unique identification designating the block number.
        # Input shape
        4D tensor with shape: `(batch, channels, rows, cols)` if
        data_format='channels_first'
        or 4D tensor with shape: `(batch, rows, cols, channels)` if
        data_format='channels_last'. # Output shape
        4D tensor with shape: `(batch, filters, new_rows, new_cols)` if
        data_format='channels_first'
        or 4D tensor with shape: `(batch, new_rows, new_cols, filters)` if
        data_format='channels_last'. `rows` and `cols` values might have
        changed due to stride.
    Returns:
        Output tensor of block.
    """
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    if strides == (1, 1):
        x = inputs
    else:
        x = ZeroPadding2D(((0, 1), (0, 1)), name='conv_pad_%d' % block_id)(inputs)
    x = DepthwiseConv2D((3, 3),
                                padding='same' if strides == (1, 1) else 'valid',
                                depth_multiplier=depth_multiplier,
                                strides=strides,
                                use_bias=False,
                                name='conv_dw_%d' % block_id)(
                                    x)
    x = BatchNormalization(
        axis=channel_axis, name='conv_dw_%d_bn' % block_id)(
            x)
    #   x = layers.ReLU(6., name='conv_dw_%d_relu' % block_id)(x)
    x = Activation(activation, name=f'conv_dw_{block_id}_{activation}')(x) 
    if noise:
        x = GaussianNoise(noise)(x)

    if bottleneck and block_id==1:
         x = Conv2D(
            4, (1, 1),
            padding='same',
            use_bias=False,
            strides=(1, 1),
            name='conv_pw_%d' % block_id)(
                x)
    else:
        x = Conv2D(
            pointwise_conv_filters, (1, 1),
            padding='same',
            use_bias=False,
            strides=(1, 1),
            name='conv_pw_%d' % block_id)(
                x)
    x = BatchNormalization(
        axis=channel_axis, name='conv_pw_%d_bn' % block_id)(
            x)
    x = Activation(activation, name=f'conv_pw_{block_id}_{activation}')(x) #layers.ReLU(6., name='conv1_relu')(x)
    if noise:
        x = GaussianNoise(noise)(x)

    return x #layers.ReLU(6., name='conv_pw_%d_relu' % block_id)(x)