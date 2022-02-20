import config
from unet import *
from utils import *
from losses import *
from metrics import *
from cross_validate import cross_validation
import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


if __name__ == '__main__':
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


