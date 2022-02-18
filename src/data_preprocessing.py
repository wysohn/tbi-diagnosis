import os
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import sklearn
import h5py
from tensorflow.keras import backend as K
from scipy.io import loadmat
from multiprocessing import Process, Lock, Manager
from PIL import Image

data_dir = "/DATA/phan92/tbi_diagnosis/data/raw/cardiac_displacement_3_updated_Oct13"

old_bad_patients = [1, 14, 22, 23, 27, 28, 32, 34, 35, 36, 37, 38, 39, 44, 49, 69, 71, 
                78, 82, 90, 98, 101, 121, 124, 128, 133, 928]

IPH_patients = ['DoD008', 'DoD009', 'DoD010', 'DoD012', 'DoD022', 'DoD047', 'DoD053', 
                'DoD062', 'DoD066', 'DoD067', 'DoD069', 'DoD074', 'DoD075', 'DoD078', 
                'DoD085', 'DoD089', 'DoD093', 'DoD101', 'DoD105', 'DoD107', 'DoD110', 
                'DoD112', 'DoD113', 'DoD120', 'DoD121', 'DoD126', 'DoD129', 'DoD130', 
                'DoD133']

bad_patients = ['DoD027', 'DoD028', 'DoD035', 'DoD036', 'DoD038', 'DoD049', 'DoD069', 'DoD090']

# Extract axis information to produce cone-shape images
def extract_axis(datapath, axisPath):
    data = loadmat(datapath)

    xaxis = np.array(list(data['xAxis']))
    yaxis = np.array(list(data['zAxis']))

    xaxis = cv2.resize(xaxis, (80, 256), interpolation=cv2.INTER_AREA)
    yaxis = cv2.resize(yaxis, (80, 256), interpolation=cv2.INTER_AREA)

    xaxis += 100
    yaxis -= 4

    print("saved axis info in : {}".format(axisPath))
    np.save(os.path.join(axisPath, "xAxis.npy"), xaxis)
    np.save(os.path.join(axisPath, "yAxis.npy"), yaxis)
    
    return xaxis, yaxis