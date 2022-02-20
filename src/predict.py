import config
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
from tensorflow.keras.metrics import (
    Recall,
    Precision
)
from tensorflow.keras.layers import concatenate
from tensorflow.keras.optimizers import Adam
from keras import Model
from statistics import mean, pstdev


def visualize_result(label, prediction, displacement, name, xAxis, yAxis):
    """
    Display an input
    
    Args:
        label (numpy.arr): the label
        prediction (numpy.arr): the prediction
        displacement (numpy.arr): displacement
        name (string): name of the data file
        xAxis (numpy.arr): numpy array contain x axis for display the cone
        yAxis (numpy.arr): numpy array contain y axis for display the cone
        path (string): path to save image to
    """
    label = np.squeeze(label, axis=-1)
    prediction = np.squeeze(np.squeeze(prediction, axis=-1), axis=0)
    displacement = np.squeeze(displacement, axis=-1)
    
    name = name.split('.')[0]
    
    # display
    fig, ax = plt.subplots(1, 3, figsize=(24, 6))
    fig.patch.set_facecolor('white')
    fig.suptitle(name, fontsize=16)
    # label
    ax[1].pcolormesh(xAxis, -yAxis, label, shading='auto', cmap='magma')
    ax[1].title.set_text("Ground Truth")
    ax[1].axis('off')
    # prediction
    ax[2].pcolormesh(xAxis, -yAxis, prediction, shading='auto', cmap='magma')
    ax[2].title.set_text("Prediction")
    ax[2].axis('off')
    # displacement
    ax[0].pcolormesh(xAxis, -yAxis, displacement, shading='auto')
    ax[0].title.set_text("Standardized Displacement")
    ax[0].axis('off')
    plt.show()


def show_prediction(model, dataset, sample_num):
    x = dataset['x']
    y = dataset['y']
    names = dataset['filename']
    name = names[sample_num]
    sample = x[sample_num]
    y_true = one_hot_raw(y[sample_num])
    y_pred = model.predict(np.expand_dims(sample, axis=0))
    visualize_result(y_true, y_pred, sample, name.decode('utf-8'), xAxis, yAxis)


def show_specific_prediction(file_name:str, data_dir, model):
    """
    Make a prediction for a specific file
    
    Args:
        file_name: name of the .mat file (utf-8)
        data_dir: location of the saved dataset
        model: a trained model
    """
    data = h5py.File(data_dir)
    name_ascii = file_name.encode('ascii')
    canPredict = find_and_predict(data['test'], name_ascii, model)
    if not canPredict:
        canPredict = find_and_predict(data['dev'], name_ascii, model)
    data.close()
    if not canPredict:
        print(file_name, "not found")


def find_and_predict(dataset, file_name, model):
    """
    Make a prediction for a specific file
    
    Args:
        file_name: name of the .mat file (ascii)
        dataset: train or test data set (h5py dataset)
        model: a trained model
    
    Returns:
        True if the file exist, False otherwise
    """
    names = np.array(dataset['filename'])
    idx = np.where(names == file_name)[0]
    if idx.shape[0] > 0:
        idx = idx[0]
        show_prediction(model, dataset, idx)
        return True
    
    return False


def make_and_save_prediction(model, data_dir, save_dir):
    """
    Make a prediction for a sample using a trained model
    and save the result image to a file
    
    Args:
        model
        data_dir
        save_dir
        save_name
    """
    f = h5py.File(data_dir, 'r')
    test = f['test']
    x = np.array(test['x'])
    y = one_hot_raw(test['y'])
    names = list(test['filename'])
    f.close()
    
    y_pred = model.predict(x)
    
    for idx in range(x.shape[0]):
        name = names[idx].decode('utf-8')
        name = name.split('.')[0]
        path = os.path.join(save_dir, name + ".png")

        label = np.squeeze(y[idx], axis=-1)
        prediction = np.squeeze(y_pred[idx], axis=-1)
        displacement = np.squeeze(x[idx], axis=-1)
        # display
        fig, ax = plt.subplots(1, 3, figsize=(24, 6))
        fig.patch.set_facecolor('white')
        fig.suptitle(name, fontsize=16)

        # label
        ax[1].pcolormesh(xAxis, -yAxis, label, shading='auto', cmap='magma')
        ax[1].title.set_text("Ground Truth")
        ax[1].axis('off')
        # prediction
        ax[2].pcolormesh(xAxis, -yAxis, prediction, shading='auto', cmap='magma')
        ax[2].title.set_text("Prediction")
        ax[2].axis('off')
        # displacement
        ax[0].pcolormesh(xAxis, -yAxis, displacement, shading='auto')
        ax[0].title.set_text("Standardized Displacement")
        ax[0].axis('off')
        plt.savefig(path)


if __name__ == '__main__':
    save_dir = os.path.join(config.INFERENCE_DIR, '02192022-skull_hybrid_32_filters_unet_with_attention')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    make_and_save_prediction(model, hdf5_dir, save_dir)