import config
from unet import *
from utils import *
from losses import *
from metrics import *
from cross_validate import cross_validation
import sys
import os
import numpy as np
from tensorboard.plugins.hparams import api as hp
from sklearn.model_selection import train_test_split

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# setup hyperparameter experiment
HP_FILTERS = hp.HParam('filters', hp.Discrete([8, 16, 32]))
HP_BATCH_SIZE = hp.HParam('batch size', hp.Discrete([10, 20, 30, 32]))
HP_DROPOUT = hp.HParam('drop out', hp.Discrete([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd', 'RMSprop']))
METRIC = 'dice_coefficient'

with tf.summary.create_file_writer('/DATA/phan92/hparam_tuning').as_default():
    hp.hparams_config(
        hparams=[HP_FILTERS, HP_BATCH_SIZE, HP_DROPOUT, HP_OPTIMIZER],
        metrics=[hp.Metric(METRIC, display_name='Dice')]
    )


def train_test_model(hdf5_file, hparams):
    """
    Helper function for parameter tuning
    :param df: input data
    :param hparams: set of hyperparameter
    :return: evaluation metric
    """
    # get data
    f = h5py.File(hdf5_file, 'r')
    dev = f['dev']
    x = np.array(dev['x'])
    y = one_hot_float(dev['y'])

    # split the dataset
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    model = create_segmentation_model(256, 80,
                                      filters = hparams[HP_FILTERS], 
                                      architecture='unet', 
                                      level = 4, 
                                      dropout_rate=hparams[HP_DROPOUT])

    model.compile(optimizer=hparams[HP_OPTIMIZER], 
                    loss=soft_dice_loss(0.00001), 
                    metrics=[dice_coefficient])

    # train the model
    model.fit(X_train, Y_train, batch_size=hparams[HP_BATCH_SIZE], epochs=30)
    # evaluate
    _, dice = model.evaluate(X_test, Y_test)

    f.close()

    return dice


def run(run_dir, hparams, hdf5_file):
    """
    Helper function for hyperparameter tuning
    :param run_dir: directory to store results in
    :param hparams: individual hyperparameter to execute on
    :param df: input data
    :return: None
    """
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        dice = train_test_model(hdf5_file, hparams)
        tf.summary.scalar(METRIC, dice, step=1)


def grid_search(hdf5_file, log_dir):
    """
    Tune hyperparameters to select deep learning model
    :param df: input data frame containing raw data
    :return: None
    """
    session_num = 0

    for filters in HP_FILTERS.domain.values:
        for batch_size in HP_BATCH_SIZE.domain.values:
            for dropout_rate in HP_DROPOUT.domain.values:
                for optimizer in HP_OPTIMIZER.domain.values:
                    hparams = {
                        HP_FILTERS: filters,
                        HP_BATCH_SIZE: batch_size,
                        HP_DROPOUT: dropout_rate,
                        HP_OPTIMIZER: optimizer
                    }
                    run_name = "run-%d" % session_num
                    print('--- Starting trial: %s' % run_name)
                    print({h.name: hparams[h] for h in hparams})
                    run(os.path.join(log_dir, run_name), hparams, hdf5_file)
                    session_num += 1


if __name__ == '__main__':
    '''
    # hyperparameters
    start_filters = [8, 16, 32]
    batch_sizes = [10, 20, 30, 32]
    n_epochs = [30]
    dropouts = [0, 0.3, 0.4, 0.5]
    loss_fns = ["soft_dice_loss", "wbce", "hybrid"]

    sys.stdout = open("parameter_tuning.txt", "w")

    # tune model using the selected hyperparameters
    data_dir = os.path.join(config.PROCESSED_DATA_DIR, "skull_displacementNorm_data.hdf5")
    for epochs in n_epochs:
        for filters in start_filters:
            for batch_size in batch_sizes:
                for dropout in dropouts:
                    for loss_fn in loss_fns:
                        cross_validation(batch_size, filters, dropout, loss_fn, epochs, 10, data_dir)
    
    sys.stdout.close()
    '''
    data_dir = os.path.join(config.PROCESSED_DATA_DIR, "skull_displacementNorm_data.hdf5")
    log_dir = '/DATA/phan92/hparam_tuning'
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    grid_search(data_dir, log_dir)

