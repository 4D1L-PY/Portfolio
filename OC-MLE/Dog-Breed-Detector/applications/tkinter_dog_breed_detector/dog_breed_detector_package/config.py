import os
import sys
import pickle
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf


def resource_path(relative_path):
    """ 
    Get absolute path to resource, works for dev and for PyInstaller
    
    :param relative_path: relative file path

    :return: absolute file path
    """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


# Dog breeds detection model image size
IMG_SIZE = (224, 224)

# Get absolute path to breeds labels & model
breeds_labels_filename = resource_path('data_files/dog_breeds_labels.pkl')
breed_detector_filename = resource_path('data_files/dog_breed_detector.h5')

# Load breeds labels
BREEDS = pickle.load(open(breeds_labels_filename, 'rb')) 
# Load dog breeds detector trained model
MODEL = tf.keras.models.load_model(breed_detector_filename)

# Tkinter configuration
BUTTONS_FONT = 'times 10 bold'
PREDICTIONS_FONT = 'times 12 bold'
