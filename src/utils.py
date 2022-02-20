import cv2
import os
import numpy as np
from scipy.io import loadmat


# Extract axis information to produce cone-shape images
def extract_axis(datapath, axisPath):
    """
    Extract axis information to produce cone-shape images

    Args:
        datapath: string: path to a raw .mat file
        axisPath: string: path to store the axis
    
    Returns:
        xaxis
        yaxis
    """
    data = loadmat(datapath)

    xaxis = np.array(list(data['xAxis']))
    yaxis = np.array(list(data['zAxis']))

    xaxis = cv2.resize(xaxis, (80, 256), interpolation=cv2.INTER_AREA)
    yaxis = cv2.resize(yaxis, (80, 256), interpolation=cv2.INTER_AREA)

    xaxis += 100
    yaxis -= 4

    print("saved axis info in : {}".format(axisPath))
    np.save(os.path.join(axisPath, "xAxis.npy"), xaxis)
    np.save(os.path.join(axisPath, "yAxis.npy"), yaxis)
    
    return xaxis, yaxis


def one_hot_binary(label, objective='skull'):
    """
    Create one hot label from the provided label
    The output has probability converted to int
    
    Args:
        label: label of shape (N, x_dim, y_dim)
        objective (string): objective of the model. brain, skull, or bleed
        
    Returns:
        label: onehot label (N, x_dim, y_dim, n_channels)
    """
    label = np.array(label)
    if objective == 'brain':
        # create label to find brain tissue
        # if the probability of brain is > 0, set the label to 1, 0 otherwise
        label = np.where(label > 0, 1, 0)
    elif objective == 'skull':
        # create label to find the skull
        # if the probability of skull is >= 0.5
        # set the label to 1, 0 otherwise
        label = np.where(label >= 0.5, 1, 0)
    elif objective == 'bleed':
        # create label to find bleed
        # if the probability of bleed is >= 0.3
        # set the label to 1
        label = np.where(label >= 0.3, 1, 0)
    else:
        raise ValueError("No objective with name \"{name}\"".format(name=objective))
    
    # add a n_channel dim
    label = np.expand_dims(label, axis=-1)
    #label = np.moveaxis(label, 3, 1)
    
    return label


def one_hot_float(label):
    """
    Create one hot label from the provided label
    
    Args:
        label: label of shape (N, x_dim, y_dim, n_classes)
        objective (string): objective of the model. brain, skull, or bleed
        
    Returns:
        label: onehot label (N, x_dim, y_dim, n_channels)
    """
    # add a n_channel dim
    label = np.expand_dims(label, axis=-1)
    
    return label