import numpy as np
import os
import config
import random
import math
import cmath
import time
import matplotlib
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import backend as K 
from sklearn.model_selection import KFold
import h5py
from datetime import datetime
from tensorflow.python.keras.callbacks import (
    ModelCheckpoint, 
    TensorBoard, 
    LambdaCallback,
    ReduceLROnPlateau,
    EarlyStopping
)
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
from tensorflow.keras import Model
from statistics import mean, pstdev
from unet import *
from losses import *
from metrics import *
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

architecture = config.MODEL_TYPE


def train(model, stage, hdf5_file: str, checkpoint_dir: str, log_dir: str, batch_size:int, epochs=50):
    """
    Trains UNet model with data contained in given HDF5 file and saves
    trained model to the checkpoint directory after each epoch
    
    Args:
        model: a complied model
        stage: 'validate' or 'test'. denote validating on 20% of training set
            or testing on unseen data
        hdf5_file: Path of hdf5 file which contains the dataset
        checkpoint_dir (str): Directory where checkpoints are saved
        log_dir (str): Directory where logs are saved
        epochs (int): number of epochs
    """
    dataset = h5py.File(hdf5_file, 'r')
    
        # make dir for tensorboard logging
    log_dir = os.path.join(log_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir)
    
    # callback to save trained model weights
    checkpoint_dir = os.path.join(checkpoint_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(checkpoint_dir)
    weights_file = os.path.join(checkpoint_dir, architecture + '.weights.epoch_{epoch:02d}.hdf5')
    
    # callback list
    callback_list = [
        TensorBoard(
            log_dir=log_dir, 
            write_images=True
        ),
        ModelCheckpoint(
            weights_file, 
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=5
        ),
    ]
    
    # test or validate
    if stage == 'test':
        training_generator = DataGenerator(dataset['dev'], batch_size)
        validation_generator = DataGenerator(dataset['test'], batch_size)
        model.fit(
            training_generator,
            validation_data=validation_generator,
            batch_size=batch_size,
            callbacks=callback_list,
            epochs=epochs
        )
    
    # test or validate
    if stage == 'validate':
        # get data
        dev = dataset['dev']
        x = np.array(dev['x'])
        y = one_hot_float(dev['y'])

        # split the dataset
        X_train, X_val, Y_train, Y_val = train_test_split(x, y, test_size=0.2, random_state=0)

        model.fit(X_train, Y_train,
                validation_data=(X_val, Y_val),
                batch_size=batch_size,
                callbacks=callback_list,
                epochs=epochs)

    dataset.close()


def train_cascade(model, stage, hdf5_file: str, checkpoint_dir: str, log_dir: str, batch_size:int, epochs=50):
    """
    Train cascade model
    """
    dataset = h5py.File(hdf5_file, 'r')
    
    # make dir for tensorboard logging
    log_dir = os.path.join(log_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir)
    
    # callback to save trained model weights
    checkpoint_dir = os.path.join(checkpoint_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(checkpoint_dir)
    weights_file = os.path.join(checkpoint_dir, architecture + '.weights.epoch_{epoch:02d}.hdf5')
    
    # callback list
    callback_list = [
        TensorBoard(
            log_dir=log_dir, 
            write_images=True
        ),
        ModelCheckpoint(
            weights_file, 
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=5
        )
    ]
    
    # validate or test
    if stage == 'test':
        training_generator = Cascade_DataGenerator(dataset['dev'], batch_size)
        validation_generator = Cascade_DataGenerator(dataset['test'], batch_size)
        model.fit(training_generator,
                validation_data=validation_generator,
                batch_size=batch_size,
                callbacks=callback_list,
                epochs=epochs)
    
    # validate or test
    if stage == 'validate':
        # get data
        dev = dataset['dev']
        x = np.array(dev['x'])
        y = one_hot_float(dev['y'])
        ROI = np.array(dev['ROI'])

        # split the dataset
        X_train, X_val, Y_train, Y_val = train_test_split(x, y, test_size=0.2, random_state=0)
        ROI_train, ROI_val = train_test_split(ROI, test_size=0.2, random_state=0)

        model.fit([X_train, ROI_train], Y_train,
                validation_data=([X_val, ROI_val], Y_val),
                batch_size=batch_size,
                callbacks=callback_list,
                epochs=epochs)

    dataset.close()


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

    # if architecture == 'cascade_unet_conv' or architecture == 'cascade_unet_concat':
    #     dataFile = objective + '_cascade_displacementNorm_data.hdf5'
    # else:
    #     dataFile = objective + '_displacementNorm_data.hdf5'

    architecture = config.MODEL_TYPE
    dataFile = config.TARGET_FILE
    hdf5_dir = os.path.join(config.PROCESSED_DATA_DIR, dataFile)
    batch_size = config.BATCH_SIZE
    epochs = config.EPOCHS

    print("Training for with data", dataFile, "and model", architecture)
    K.clear_session()
    model = create_segmentation_model(input_height=256,
                                      input_width=80, 
                                      filters=32, 
                                      architecture=architecture, 
                                      level=4,
                                      dropout_rate=0.3)

    plot_model(model)

    model.compile(
        optimizer=Adam(learning_rate=0.001), 
        loss=soft_dice_loss(),
        #loss=combo_loss(alpha=0.7, beta=0.5),
        metrics=[dice_coefficient, iou, Recall(), Precision()]
    )

    # get the stage of the training from console: 
    # enter 'validate' to validate on 20% of the training set
    # enter 'test' to test on unseen data
    stage = input("Enter training mode ('validate' or 'test'): ")

    if architecture == 'cascade_unet_conv' or architecture == 'cascade_unet_concat':
        train_cascade(model, 
            stage=stage,
            hdf5_file=hdf5_dir,
            checkpoint_dir=config.CHECKPOINT_DIR,
            log_dir=config.TENSORFLOW_LOG_DIR,
            batch_size=batch_size,
            epochs=epochs
        )
    else:
        train(model, 
            stage=stage,
            hdf5_file=hdf5_dir,
            checkpoint_dir=config.CHECKPOINT_DIR,
            log_dir=config.TENSORFLOW_LOG_DIR,
            batch_size=batch_size,
            epochs=epochs
        )
    
    model_saved_name = datetime.now().strftime('%Y%m%d-%H%M%S') + '_' + architecture + '_' + objective +'.h5'
    save_path = os.path.join(config.TRAINED_MODELS_DIR, model_saved_name)

    model.save(save_path)
    print('Saved trained model to', save_path)
