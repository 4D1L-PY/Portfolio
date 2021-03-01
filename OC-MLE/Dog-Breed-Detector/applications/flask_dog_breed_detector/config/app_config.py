import os
import pickle
import tensorflow as tf
# from src.object_detection import ObjectDetector

########################################################################################################## 
#                                         FILENAMES & PATHS                                              #
########################################################################################################## 

# Main logo
APP_LOGO_NAME = 'dog_breeds_detector_logo.png'

# Image uploader button logo
UPLOAD_BUTTON_LOGO_NAME = 'dog_breeds_detector_button_logo.svg'

# Main directory
CWD = os.getcwd()

# Main dog images folder
UPLOAD_FOLDER = os.path.join(CWD, 'static/img/dogs/')

# Serveur configuration
SERVER = {'HOST': '0.0.0.0', 'PORT': 8080}

########################################################################################################## 
#                                  PREPROCESSING DATA & MODEL                                            #
########################################################################################################## 

# Dog breeds detection model image size
IMG_SIZE = (224, 224)

# Load breed labels
BREEDS = pickle.load(open(os.path.join(CWD,'data/dependency/dog_breeds_labels.pkl'), 'rb')) 

# Load dog breeds detector trained model
MODEL = tf.keras.models.load_model(os.path.join(CWD,'data/model/dog_breed_detector.h5')) 
# TF LITE version : tf.lite.Interpreter(model_path=os.path.join(CWD, 'data/model/pruned_and_quantized_dog_breed_detector.tflite'))

# Optional : object detection model (only if the server has a GPU)
# object_detector = ObjectDetector(module='inception_resnet_v2')
