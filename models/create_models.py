""" Create model architectures for training. """
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.layers import *
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
from tensorflow.keras.applications import densenet
from tensorflow_addons import optimizers as tfa_optimizers

def _init_meta_model(weights=None):
    """Creates metadata model.
    
    Args:
        weights: Filename.
            Deafult is None. 
            If provided, function will attempt to initialize model weights from this file.
            Layer names in the weights file need to match exactly.

    Returns: Uncompiled tf.keras.models.Model instance.      
    """
    inp_meta = Input(shape=(22, ),name='meta_0')
    meta_x = Dense(units=32, activation='relu', name='meta_1')(inp_meta)
    meta_out = Dense(units=1, activation='sigmoid', name='meta_out')(meta_x)

    nn_mdl = models.Model(inputs=inp_meta, outputs=meta_out)
    if weights:
        nn_mdl.load_weights(weights, by_name=True)

    return nn_mdl

def _init_image_model(densenet_weights='imagenet', weights=None):
    """Creates image model.
    
    Args:
        weights: Filename.
            Deafult is None. 
            If provided, function will attempt to initialize model weights from this file.
            Layer names in the weights file need to match exactly. 
            Otherwise weights will be initialized randomly.
        densenet_weights: String.
            Either of on {'imagenet', None} or filename. Deafult is 'imagenet'. 
            If 'imagenet' model weights will be initialized to imagenet-trained weights.
            If  filename is provided, function will attempt to initialize model weights from this file.
            Layer names in the weights file need to match exactly.
            Otherwise weights will be initialized randomly.

    Returns: Uncompiled tf.keras.models.Model instance.      
    """
    densenet = densenet.DenseNet121(include_top=False, weights=densenet_weights, input_shape=(224, 224, 3), pooling='avg')

    inp_img = Input(shape=(224, 224, 2,), name='img_0')
    x_img = Conv2D(filters=3, kernel_size=(1,1), name='img_1')(inp_img)
    x_img = densenet(x_img)
    x_img = Dropout(rate=0.5, name='img_3')(x_img)
    img_out = Dense(units=1, activation='sigmoid', name='img_out')(x_img)

    nn_mdl = models.Model(inputs=[inp_img], outputs=[img_out])
    if weights:
        nn_mdl.load_weights(weights, by_name=True)

    return nn_mdl

def _init_combined_model(densenet_weights='imagenet', weights=None):
    """Creates combined model.
    
    Args:
        weights: Filename.
            Deafult is None. 
            If provided, function will attempt to initialize model weights from this file.
            Layer names in the weights file need to match exactly. 
            Otherwise weights will be initialized randomly.
        densenet_weights: String.
            Either of on {'imagenet', None} or filename. Deafult is 'imagenet'. 
            If 'imagenet' model weights will be initialized to imagenet-trained weights.
            If  filename is provided, function will attempt to initialize model weights from this file.
            Layer names in the weights file need to match exactly.
            Otherwise weights will be initialized randomly.

    Returns: Uncompiled tf.keras.models.Model instance.      
    """
    densenet = densenet.DenseNet121(include_top=False, weights=densenet_weights, input_shape=(224, 224, 3), pooling='avg')
    
    inp_meta = Input(shape=(22, ),name='meta_0')
    x_meta = Dense(units=32, name='meta_1')(inp_meta)
    
    inp_img = Input(shape=(224, 224, 2,), name='img_0')
    x_img = Conv2D(filters=3, kernel_size=(1,1), name='img_1')(inp_img) 
    x_img = densenet(x_img)

    comb_inp = Concatenate(name='comb_0')([x_img, x_meta])

    comb_x = Dense(units=512, activation='relu', name='comb_1')(comb_inp)
    comb_x = Dense(units=64, activation='relu', name='comb_2')(comb_x)
    comb_x = Dropout(rate=0.5, name='comb_3')(comb_x)

    comb_out = Dense(units=1, activation='sigmoid', name='comb_4')(comb_x)

    nn_mdl = models.Model(inputs=[inp_img, inp_meta], outputs=[comb_out])
    if weights:
        nn_mdl.load_weights(weights, by_name=True)

    return nn_mdl