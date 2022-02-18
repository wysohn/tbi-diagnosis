import h5py
import numpy as np
import tensorflow as tf
from utils import *

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
        y = one_hot_float(self.y_dset[batch_start : batch_end])
        
        return x, y