import os
import dotenv

dotenv.load_dotenv()

CUDA_VISIBLE_DEVICES = str(os.getenv("CUDA_VISIBLE_DEVICES"))

# Path to raw data
RAW_DATA_DIR = str(os.getenv("RAW_DATA_DIR"))

# Path to processed numpy matrices (where hdf5 files are stored)
PROCESSED_DATA_DIR = str(os.getenv("PROCESSED_DATA_DIR"))

# Name of the target hdf5 file to be used for training/testing
TARGET_FILE = str(os.getenv("TARGET_FILE"))

# Name of the model file in the TRAINED_MODELS_DIR to be used for performance evaluation
MODEL_NAME = str(os.getenv("MODEL_NAME"))

# Path to trained models
TRAINED_MODELS_DIR = str(os.getenv("TRAINED_MODELS_DIR"))

# Path to saved inference results
INFERENCE_DIR = str(os.getenv("INFERENCE_DIR"))

# Path to tensorflow training logs
TENSORFLOW_LOG_DIR = str(os.getenv("TENSORFLOW_LOG_DIR"))

# path to training check point
CHECKPOINT_DIR = str(os.getenv("CHECKPOINT_DIR"))

# objective of the model
# essentially the target for the model to detect:
    #   mode 0 = skull
    #   mode 1 = blood
    #   mode 2 = brain
    #   mode 3 = ventricle
DATA_MODE = int(os.getenv('DATA_MODE')) if os.getenv('DATA_MODE') else None

# where to store hyperparameter experiments
HYPERPARAM = str(os.getenv('HYPERPARAM'))

# hyper parameters
BATCH_SIZE = int(os.getenv('BATCH_SIZE')) if os.getenv('BATCH_SIZE') else None

EPOCHS = int(os.getenv('EPOCHS')) if os.getenv('EPOCHS') else None

DROPOUT_RATE = float(os.getenv('DROPOUT_RATE')) if os.getenv('DROPOUT_RATE') else None

NUM_FILTERS = int(os.getenv('NUM_FILTERS')) if os.getenv('NUM_FILTERS') else None

# the type of model
    # unet
    # unet_plus_plus
    # attention_unet
    # cascade_unet_conv
    # cascade_unet_concat
MODEL_TYPE = str(os.getenv('MODEL_TYPE'))

# if using fullly cascaded model to make prediction
cascade = str(os.getenv('FULL_CASCADE'))
FULL_CASCADE = False
if cascade == 'yes':
    FULL_CASCADE = True
