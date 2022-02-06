import numpy as np
import os
import random
import math
import cmath
import multiprocessing
import random
import time
import matplotlib
import keras
import tensorflow as tf
from tensorflow.keras import backend as K 
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
import h5py
from datetime import datetime
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard, LambdaCallback
from sklearn.metrics import confusion_matrix
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Activation,
    Conv2D,
    Conv2DTranspose,
    MaxPooling2D,
    UpSampling2D,
    Dropout
)
from tensorflow.keras.layers import concatenate
from tensorflow.keras.optimizers import Adam
from keras import Model


def one_hot(label, objective='skull'):
    """
    Create one hot label from the provided label
    
    Args:
        label: label of shape (N, x_dim, y_dim)
        objective (string): objective of the model. brain, skull, or bleed
        
    Returns:
        label: onehot label (N, x_dim, y_dim, n_channels)
    """
    if objective == 'brain':
        # create label to find brain tissue
        # if the probability of brain is > 0, set the label to 1, 0 otherwise
        label = np.where(label > 0, 1, 0)
    elif objective == 'skull':
        # create label to find the skull
        # if the probability of skull is >= 0.5
        # set the label to 1, 0 otherwise
        label = np.where(label >= 1.5, 1, 0)
    elif objective == 'bleed':
        # create label to find bleed
        # if the probability of bleed is >= 0.3
        # set the label to 1
        label = np.where(label >= 2.3, 1, 0)
    else:
        raise ValueError("No objective with name \"{name}\"".format(name=objective))

    label = np.expand_dims(label, axis=3)
    #label = np.moveaxis(label, 3, 1)
    
    return label.astype(int16)

class DataGenerator(tf.keras.utils.Sequence):
    """
    Read saved data and provide it to the model in batches
    """
    
    def __init__(self, group: h5py.Group, batch_size=32):
        self.x_dset: h5py.Dataset = group['x']
        self.y_dset: h5py.Dataset = group['y']
        self.batch_size = batch_size
    
    
    def __len__(self):
        """
        Get the numner of batches per epoch
        """
        return int(np.floor(self.x_dset.shape[0] / self.batch_size))
    
    
    def on_epoch_end(self):
        pass
    
    
    def __getitem__(self, batch_index:int):
        """
        Generate one batch of data
        :param batch_index: index of this batch
        :return: the data x and label y of the batch
        """
        batch_start = batch_index * self.batch_size
        batch_end = batch_start + self.batch_size

        x = self.x_dset[batch_start : batch_end]
        y = one_hot(self.y_dset[batch_start : batch_end])
        
        return x, y