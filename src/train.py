import numpy as np
import os
import config
import random
import math
import cmath
import time
import matplotlib
import keras
import tensorflow as tf
from tensorflow.keras import backend as K 
from sklearn.model_selection import KFold
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
from unet import *
from losses import *
from metrics import *
from tensorflow.keras.utils import plot_model

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train(model, hdf5_file: str, checkpoint_dir: str, log_dir: str, epochs=50):
        """
        Trains UNet model with data contained in given HDF5 file and saves
        trained model to the checkpoint directory after each epoch
        
        Args:
            hdf5_file: Path of hdf5 file which contains the dataset
            checkpoint_dir (str): Directory where checkpoints are saved
            log_dir (str): Directory where logs are saved
            epochs (int): number of epochs
        """
        dataset = h5py.File(hdf5_file, 'r')
        
        training_generator = DataGenerator(dataset['dev'])
        validation_generator = DataGenerator(dataset['test'])
        
        # callback for tensorboard logs
        log_dir = os.path.join(log_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(log_dir)
        tb_callback = TensorBoard(log_dir=log_dir, write_images=True)
        
        # callback to save trained model weights
        checkpoint_dir = os.path.join(checkpoint_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(checkpoint_dir)
        weights_file = os.path.join(checkpoint_dir, 'unet.weights.epoch_{epoch:02d}.hdf5')
        checkpoint = ModelCheckpoint(weights_file, verbose=1)
        
        model.fit(training_generator,
                  validation_data=validation_generator,
                  callbacks=[tb_callback, checkpoint],
                  epochs=epochs)
        
        dataset.close()


if __name__ == '__main__':
    # objective:
    #   mode 0 = skull
    #   mode 1 = bleed
    #   mode 2 = brain
    #   mode 3 = ventricle
    mode = config.DATA_MODE
    if mode == 0:
        objective = 'skull'
    elif mode == 1:
        objective = 'bleed'
    elif mode == 2:
        objective = 'brain'
    elif mode == 3:
        objective = 'vent'
    else:
        raise ValueError("Enter a valid mode")

    dataFile = objective + '_displacementNorm_data.hdf5'
    hdf5_dir = os.path.join(config.PROCESSED_DATA_DIR, dataFile)
    architecture = config.MODEL_TYPE
    K.clear_session()
    model = create_segmentation_model(input_height=256,
                                      input_width=80, 
                                      filters=32, 
                                      architecture=architecture, 
                                      level=4,
                                      dropout_rate=0)

    plot_model(model)

    model.compile(optimizer='adam', 
                loss=soft_dice_loss(epsilon=0.00001),
                #loss=weighted_bce(beta=5),
                #loss=hybrid_loss(beta=5, epsilon=0.00001),
                metrics=[dice_coefficient, iou, Recall(), Precision()])

    train(model, 
        hdf5_file=hdf5_dir,
        checkpoint_dir=config.CHECKPOINT_DIR,
        log_dir=config.TENSORFLOW_LOG_DIR, 
        epochs=50)
    
    model_saved_name = datetime.now().strftime('%Y%m%d-%H%M%S') + '_' + architecture + '_' + objective +'.h5'
    save_path = os.path.join(config.TRAINED_MODELS_DIR, model_saved_name)

    model.save(save_path)
    print('Saved trained model to', save_path)
