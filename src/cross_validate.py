import config
from unet import *
from utils import *
from losses import *
from metrics import *
import numpy as np
import os
import random
import random
import keras
import tensorflow as tf
from sklearn.model_selection import KFold
import h5py
from tensorflow.keras import Input, Model

from tensorflow.keras.metrics import (
    Recall,
    Precision
)
from tensorflow.python.keras.callbacks import (
    ModelCheckpoint, 
    TensorBoard, 
    LambdaCallback,
    ReduceLROnPlateau,
    EarlyStopping
)
from tensorflow.keras.optimizers import Adam
from keras import Model
from statistics import mean, pstdev

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def cross_validation(architecture,
                     batch_size, 
                     start_filters, 
                     dropout,
                     loss_fn,
                     epochs, 
                     num_folds,
                     hdf5_file):
    """
    Perform kfold cross validation on the training set
    
    Args:
        num_folds (int): k in kfold
        hdf5 (string): path to hdf5 dataset
        epochs (int): number of epochs to train for
        
    Returns:
        loss_per_fold: list of loss values
        dice_per_fold: list of dice coefficient
    """
    dice_per_fold = []
    iou_per_fold = []
    loss_per_fold = []
    recall_per_fold = []
    precision_per_fold = []
    
    # get data
    f = h5py.File(hdf5_file, 'r')
    dev = f['dev']
    x = np.array(dev['x'])
    y = one_hot_float(dev['y'])

    if config.MODEL_TYPE == 'cascade_unet_conv' or config.MODEL_TYPE == 'cascade_unet_concat':
        ROI = np.array(dev['ROI'])
    
    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=0)
    
    fold_no = 1

    print("Validating for: start filters = %d, batch size = %d, epochs = %d, dropout = %f, loss_fn = %s" % (start_filters, batch_size, epochs, dropout, loss_fn))
    
    if loss_fn == "soft_dice_loss":
        loss_fn = soft_dice_loss()
    elif loss_fn == "wbce":
        loss_fn = weighted_bce(beta=5)
    elif loss_fn == "hybrid":
        loss_fn = hybrid_loss(beta=5)
    
    # callback list
    callback_list = [
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=5
        )
    ]

    for train, test in kfold.split(x, y):
        K.clear_session()
        
        # build model
        model = create_segmentation_model(256, 80, 
                                          start_filters, 
                                          architecture=architecture, 
                                          level = 4, 
                                          dropout_rate=dropout)
        model.compile(optimizer=Adam(learning_rate=0.001), 
                      loss=loss_fn, 
                      metrics=[dice_coefficient, iou, Recall(), Precision()])
        
        print("Training for fold", fold_no, "....")

        if config.MODEL_TYPE == 'cascade_unet_conv' or config.MODEL_TYPE == 'cascade_unet_concat':
            history = model.fit([x[train], ROI[train]], 
                                y[train],
                                validation_data=([x[test], ROI[test]], y[test]),
                                batch_size=batch_size, 
                                callbacks=callback_list, 
                                epochs=epochs, 
                                verbose=1
                                )
            score = model.evaluate([x[test], ROI[test]], y[test], verbose=0)
        else:
            history = model.fit(x[train], 
                                y[train],
                                validation_data=(x[test], y[test]),
                                batch_size=batch_size, 
                                callbacks=callback_list, 
                                epochs=epochs, 
                                verbose=1
                                )
            score = model.evaluate(x[test], y[test], verbose=0)

        loss_per_fold.append(score[0])
        dice_per_fold.append(score[1])
        iou_per_fold.append(score[2])
        recall_per_fold.append(score[3])
        precision_per_fold.append(score[4])
        
        fold_no += 1
    
    f.close()

    print("10-fold cross validation result for", config.MODEL_TYPE, "with n =", start_filters,
            "dropout rate =", dropout, "batch size = ", batch_size)
    print("Mean loss = %f, stdev = %f" %(mean(loss_per_fold), pstdev(loss_per_fold)))
    print("Mean dice = %f, stdev = %f" %(mean(dice_per_fold), pstdev(dice_per_fold)))
    print("Mean iou = %f, stdev = %f" %(mean(iou_per_fold), pstdev(iou_per_fold)))
    print("Mean recall = %f, stdev = %f" %(mean(recall_per_fold), pstdev(recall_per_fold)))
    print("Mean precision = %f, stdev = %f" %(mean(precision_per_fold), pstdev(precision_per_fold)))
    print()
    print()

    return loss_per_fold, dice_per_fold, iou_per_fold, recall_per_fold, precision_per_fold


if __name__ == '__main__':
    batch_size = 30
    filters = 16
    dropout = 0.1
    loss_fn = 'soft_dice_loss'
    epochs = 100
    kfold = 10

    mode = config.DATA_MODE
    if mode == 0:
        objective = 'skull'
    elif mode == 1:
        objective = 'blood'
    elif mode == 2:
        objective = 'brain'
    elif mode == 3:
        objective = 'vent'
    else:
        raise ValueError("Enter a valid mode")

    architecture = config.MODEL_TYPE
    
    if architecture == 'cascade_unet_conv' or architecture == 'cascade_unet_concat':
        data_dir = os.path.join(config.PROCESSED_DATA_DIR, objective + '_cascade_displacementNorm_data.hdf5')
    else:
        data_dir = os.path.join(config.PROCESSED_DATA_DIR, objective + '_displacementNorm_data.hdf5')

    print("Perform 10-fold cross validation for", architecture, "with data", data_dir)
    cross_validation(architecture, batch_size, filters, dropout, loss_fn, epochs, kfold, data_dir)