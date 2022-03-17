import os
import dotenv

dotenv.load_dotenv()

# Path to raw data
RAW_DATA_DIR = str(os.getenv("RAW_DATA_DIR"))

# Path to processed numpy matrices
PROCESSED_DATA_DIR = str(os.getenv("PROCESSED_DATA_DIR"))

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
    #   mode 1 = bleed
    #   mode 2 = brain
    #   mode 3 = ventricle
DATA_MODE = int(os.getenv('DATA_MODE'))

# the type of model
    # unet
    # unet_plus_plus
    # attention_unet
MODEL_TYPE = str(os.getenv('MODEL_TYPE'))
