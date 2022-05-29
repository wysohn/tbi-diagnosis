import config
from losses import *
from metrics import *
from utils import *
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
from tensorflow.keras import backend as K 
import h5py
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import (
    Recall,
    Precision
)

# extract generic axis information
axisPath = config.PROCESSED_DATA_DIR
rand_input_file = os.path.join(config.RAW_DATA_DIR, 'DoD001','DoD001_Ter030_LC5_Displacement_Normalized_3.mat')
xAxis, yAxis = extract_axis(rand_input_file)


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
    ax[2].title.set_text("Output")
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
    y_true = one_hot_float(y[sample_num])
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


def evaluate(y_true, y_pred):
    dice_score = dice_coefficient(y_true, y_pred)
    iou_score = iou(y_true, y_pred)
    precision_m = Precision()
    precision_m.update_state(y_true, y_pred)
    recall_m = Recall()
    recall_m.update_state(y_true, y_pred)
    
    print("Mean dice score:", dice_score.numpy())
    print("Mean IoU:", iou_score.numpy())
    print("Mean precision:", precision_m.result().numpy())
    print("Mean recall:", recall_m.result().numpy())


def make_and_save_prediction(model, data_dir, save_dir):
    """
    Make a prediction for a sample using a trained model
    and save the result image to a file
    
    Args:
        model
        data_dir
        save_dir
    """

    f = h5py.File(data_dir, 'r')
    test = f['test']
    x = np.array(test['x'])
    y = one_hot_float(test['y'])
    names = list(test['filename'])

    # make prediction
    if config.MODEL_TYPE == 'cascade_unet_conv' or config.MODEL_TYPE == 'cascade_unet_concat':
        ROI = np.array(test['ROI'])
        y_pred = model.predict([x, ROI])
    else:
        y_pred = model.predict(x)
    
    f.close()

    evaluate(y, y_pred)

    for idx in range(x.shape[0]):
        name = names[idx].decode('utf-8')
        name = name.split('.')[0]
        name = name[0:17]
        path = os.path.join(save_dir, name + ".png")

        label = np.squeeze(y[idx], axis=-1)
        prediction = np.squeeze(y_pred[idx], axis=-1)
        displacement = np.squeeze(x[idx], axis=-1)
        # display
        fig, ax = plt.subplots(1, 3, figsize=(24, 6))
        fig.patch.set_facecolor('white')
        fig.suptitle(name, fontsize=18, fontweight='bold')

        # label
        ax[1].pcolormesh(xAxis, -yAxis, label, shading='auto', cmap='magma')
        ax[1].set_title("Ground Truth", fontsize=14, fontweight='bold')
        ax[1].axis('off')
        # prediction
        ax[2].pcolormesh(xAxis, -yAxis, prediction, shading='auto', cmap='magma')
        ax[2].set_title("Output", fontsize=14, fontweight='bold')
        ax[2].axis('off')
        # displacement
        ax[0].pcolormesh(xAxis, -yAxis, displacement, shading='auto')
        ax[0].set_title("Standardized Displacement", fontsize=14, fontweight='bold')
        ax[0].axis('off')
        plt.savefig(path)


def make_and_save_prediction_full_cascade(model, ROI_model, data_dir, save_dir):
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
    y = one_hot_float(test['y'])
    names = list(test['filename'])
    f.close()

    # detect the brain tissue
    ROI_pred = ROI_model.predict(x)

    # make prediction using predicted ROI mask
    y_pred = model.predict([x, ROI_pred])

    evaluate(y, y_pred)
    
    for idx in range(x.shape[0]):
        name = names[idx].decode('utf-8')
        name = name.split('.')[0]
        name = name[0:17]
        path = os.path.join(save_dir, name + ".png")

        label = np.squeeze(y[idx], axis=-1)
        prediction = np.squeeze(y_pred[idx], axis=-1)
        displacement = np.squeeze(x[idx], axis=-1)
        # display
        fig, ax = plt.subplots(1, 3, figsize=(24, 6))
        fig.patch.set_facecolor('white')
        fig.suptitle(name, fontsize=18, fontweight='bold')

        # label
        ax[1].pcolormesh(xAxis, -yAxis, label, shading='auto', cmap='magma')
        ax[1].set_title("Ground Truth", fontsize=14, fontweight='bold')
        ax[1].axis('off')
        # prediction
        ax[2].pcolormesh(xAxis, -yAxis, prediction, shading='auto', cmap='magma')
        ax[2].set_title("Output", fontsize=14, fontweight='bold')
        ax[2].axis('off')
        # displacement
        ax[0].pcolormesh(xAxis, -yAxis, displacement, shading='auto')
        ax[0].set_title("Standardized Displacement", fontsize=14, fontweight='bold')
        ax[0].axis('off')
        plt.savefig(path)


if __name__ == '__main__':
    # # objective:
    # #   mode 0 = skull
    # #   mode 1 = blood
    # #   mode 2 = brain
    # #   mode 3 = ventricle
    # mode = config.DATA_MODE
    # if mode == 0:
    #     objective = 'skull'
    # elif mode == 1:
    #     objective = 'blood'
    # elif mode == 2:
    #     objective = 'brain'
    # elif mode == 3:
    #     objective = 'vent'
    # else:
    #     raise ValueError("Enter a valid mode")

    # name of a trained model
    model_name = config.MODEL_NAME
    model_path = os.path.join(config.TRAINED_MODELS_DIR, model_name)
    model = load_model(model_path, compile=False)

    dataFile = config.TARGET_FILE
    data_dir = os.path.join(config.PROCESSED_DATA_DIR, dataFile)

    # if architecture == 'cascade_unet_conv' or architecture == 'cascade_unet_concat':
    #     data_dir = os.path.join(config.PROCESSED_DATA_DIR, objective + '_cascade_displacementNorm_data.hdf5')
    # else:
    #     data_dir = os.path.join(config.PROCESSED_DATA_DIR, objective + '_displacementNorm_data.hdf5')

    print("Making prediction using model", model_name, "with data", data_dir)

    if config.FULL_CASCADE:
        print("Full cascade = yes")
        ROI_model_name = '20220426-235354_unet_plus_plus_brain.h5'
        ROI_model_path = os.path.join(config.TRAINED_MODELS_DIR, ROI_model_name)
        ROI_model = load_model(ROI_model_path, compile=False)
        save_dir = os.path.join(config.INFERENCE_DIR, 
                                datetime.now().strftime('%Y%m%d-%H%M%S') + '_full_' + dataFile)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        make_and_save_prediction_full_cascade(model, ROI_model, data_dir, save_dir)
    else:
        print("Full cascade = no")
        save_dir = os.path.join(config.INFERENCE_DIR, 
                            datetime.now().strftime('%Y%m%d-%H%M%S') + '_' + dataFile)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        make_and_save_prediction(model, data_dir, save_dir)
