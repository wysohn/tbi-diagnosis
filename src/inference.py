import os
import numpy as np
import config
from utils import *
from data_preprocessing import *
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

skull_model_path = os.path.join(config.TRAINED_MODELS_DIR, '20220425-221917_unet_skull.h5')
vent_model_path = os.path.join(config.TRAINED_MODELS_DIR, '20220427-192410_cascade_unet_concat_vent.h5')
blood_model_path = os.path.join(config.TRAINED_MODELS_DIR, '20220427-184611_cascade_unet_concat_blood.h5')
brain_model_path = os.path.join(config.TRAINED_MODELS_DIR, '20220426-235354_unet_plus_plus_brain.h5')


def preprocess(input_path, x_dim, y_dim, objective):
    """
    Prepare the input
    Args:
        input_path: the path to input displacement matrix
    Return: processed displacement frame
    """
    displacement, _, _ = process_one_file(input_path, x_dim, y_dim, objective)
    displacement = np.expand_dims(displacement, axis=0)

    return displacement


def detect(objective, displacement):
    """
    Detect the structure of interest
    Args:
        objective: skull(0), ventricle(1), or blood(2)
        input: processed displacement frame
    Return: mask for the detected structure
    """
    if objective == 'skull':
        model = load_model(skull_model_path, compile=False)
        y_pred = model.predict(displacement)
    elif objective == 'ventricle':
        ROI_model = load_model(brain_model_path, compile=False)
        model = load_model(vent_model_path, compile=False)
        ROI = ROI_model.predict(displacement)
        y_pred = model.predict([displacement, ROI])
    elif objective == 'blood':
        ROI_model = load_model(brain_model_path, compile=False)
        model = load_model(blood_model_path, compile=False)
        ROI = ROI_model.predict(displacement)
        y_pred = model.predict([displacement, ROI])
    else:
        raise ValueError("Enter a valid objective")

    return y_pred


def display(y_pred, input_path, save_path):
    """
    Display the output
    """
    xAxis, yAxis = extract_axis(input_path)
    plt.pcolormesh(xAxis, -yAxis, y_pred, shading='auto', cmap='magma')
    plt.savefig(os.path.join(save_path, 'output'))


if __name__ == '__main__':
    print("This program takes a displacement matrix and predict the location of a structure of interest")
    print("The detection modes are (0) Skull, (1) Ventricle, (2) Blood")
    # get the objective from the user
    mode = int(input("Enter the objective. Skull(0), ventricle(1), or blood(2): "))

    if mode == 0:
        objective = 'skull'
    elif mode == 1:
        objective = 'blood'
    elif mode == 2:
        objective = 'ventricle'
    else:
        raise ValueError("Enter a valid mode")

    input_path = input("Enter the path to input file: ")
    output_path = input("Enter the path to saved output: ")

    processed_frame = preprocess(input_path, 256, 80, objective)

    output = detect(objective, processed_frame)
    output = np.squeeze(output, axis=-1)
    output = np.squeeze(output, axis=0)

    display(
        output,
        input_path,
        output_path
    )

