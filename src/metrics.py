import tensorflow as tf
from tensorflow.keras import backend as K

def dice_coefficient(y_true, y_pred, epsilon = 0.00001):
    """
    Compute mean dice coefficient over all class (2 x overlap / total pixels)
    
    Args:
        y_true (Tensorflow tensor): tensor of ground truth
                                    shape: (x_dim, y_dim, num_class)
        y_pred (Tensorflow tensor): tensor of soft prediction for all classes
                                    shape: (x_dim, y_dim, num_class)
        epsilon (float): small constant added to avoid divide by 0 error
    Return:
        dice_coefficient (float): computed value of dice coefficient
    """
    y_true = tf.cast(y_true, tf.float64)
    y_pred = tf.cast(y_pred, tf.float64)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    total = K.sum(y_true_f) + K.sum(y_pred_f)
    dice_coeff = K.mean((2. * intersection + epsilon) / (total + epsilon))
    
    return dice_coeff


def iou(y_true, y_pred, epsilon=0.00001):
    """
    Compute mean intersection over union over all class (area of overlap / area of union)
    
    Args:
        y_true (Tensorflow tensor): tensor of ground truth
                                    shape: (x_dim, y_dim, num_class)
        y_pred (Tensorflow tensor): tensor of soft prediction for all classes
                                    shape: (x_dim, y_dim, num_class)
        epsilon (float): small constant added to avoid divide by 0 error
    Return:
        iou (float): computed value of intersection over union
    """
    y_true = tf.cast(y_true, tf.float64)
    y_pred = tf.cast(y_pred, tf.float64)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    iou = K.mean((intersection + epsilon) / (union + epsilon))
    
    return iou