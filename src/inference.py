import os
import config


skull_model_path = os.path.join(config.TRAINED_MODELS_DIR, )
vent_model_path = os.path.join(config.TRAINED_MODELS_DIR, )
blood_model_path = os.path.join(config.TRAINED_MODELS_DIR, )
brain_model_path = os.path.join(config.TRAIN_MODELS_DIR, )


def preprocess(input_path):
    """
    Prepare the input
    Args:
        input_path: the path to input displacement matrix
    Return: processed displacement frame
    """
    pass

def detect(objective, input):
    """
    Detect the structure of interest
    Args:
        objective: skull(0), ventricle(1), or blood(2)
        input: processed displacement frame
    Return: mask for the detected structure
    """
    pass


if __name__ == '__main__':
    print("This program takes a displacement matrix and predict the location of a structure of interest")
    print("The detection modes are (0) Skull, (1) Ventricle, (2) Blood")
    # get the objective from the user
    objective = input("Enter the number code for the objective:")
    input_path = input("Enter the path to input file:")

    processed_frame = preprocess(input_path)

    output = detect(objective, processed_frame)

