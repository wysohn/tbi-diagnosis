import tensorflow as tf
from tensorflow.keras import backend as K

def soft_dice_loss(epsilon=0.00001):
    """
    Compute mean soft dice loss over all classes
    Soft dice loss operate on float probability output
    The denominator is squared according to https://mediatum.ub.tum.de/doc/1395260/1395260.pdf
    
    Args:
        y_true (Tensorflow tensor): tensor of ground truth
                                    shape: (x_dim, y_dim, num_class)
        y_pred (Tensorflow tensor): tensor of soft prediction for all classes
                                    shape: (x_dim, y_dim, num_class)
        epsilon (float): small constant added to avoid divide by 0 error
        
    Return:
        dice_loss (float): computed value of dice loss
    """
    def dice_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float64)
        y_pred = tf.cast(y_pred, tf.float64)
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        dice_numerator = 2. * K.sum(y_true_f * y_pred_f) + epsilon
        dice_denominator = K.sum(y_true_f**2) + K.sum(y_pred_f**2) + epsilon
        loss = 1. - K.mean(dice_numerator / dice_denominator)
        
        return loss
    
    return dice_loss


def weighted_bce(beta):
    """
    Compute the weighted binary cross entropy loss according to
        S. Jadon, “A survey of loss functions for semantic segmentation,” 
        2020 IEEE Conference on Computational Intelligence in Bioinformatics 
        and Computational Biology (CIBCB), pp. 1–7, Oct. 2020, doi: 10.1109/CIBCB48159.2020.9277638.
    
    Args:
        beta (int): weight factor for the positive class
    Return:
        wbce (float): weighted binary cross entropy score
    """
    def wbce(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float64)
        y_pred = tf.cast(y_pred, tf.float64)
        y_true = K.clip(y_true, K.epsilon(), 1. - K.epsilon())
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
        logloss = K.mean(-(y_true * K.log(y_pred) + (1. - y_true) * K.log(1. - y_pred)))
        
        return logloss
    
    return wbce


def hybrid_loss(beta, epsilon=0.00001):
    """
    Compute a hybrid loss function which is sum of soft dice loss and weighted bce
    
    Args:
        beta (int): weight factor for the positive class
    Return:
        wbce (float): weighted binary cross entropy score
    """
    def loss(y_true, y_pred):
        # Calculate soft dice loss
        y_true = tf.cast(y_true, tf.float64)
        y_pred = tf.cast(y_pred, tf.float64)
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        dice_numerator = 2. * K.sum(y_true_f * y_pred_f) + epsilon
        dice_denominator = K.sum(y_true_f**2) + K.sum(y_pred_f**2) + epsilon
        dice_loss = 1. - K.mean(dice_numerator / dice_denominator)
        
        # Calculate weighted bce
        y_true = K.clip(y_true, K.epsilon(), 1. - K.epsilon())
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
        wbce = K.mean(-(y_true * K.log(y_pred) + (1. - y_true) * K.log(1. - y_pred)))
        
        hybrid = dice_loss + wbce
        return hybrid
    
    return loss