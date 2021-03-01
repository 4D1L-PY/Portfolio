import os
import re
import time
import schedule
import numpy as np
import tensorflow as tf
from glob import glob
from src.image_preprocessor import cv2, get_img_array, grad_cam
from config.app_config import *


def schedule_image_files_deletion(folder_path=UPLOAD_FOLDER+'*', scheduled_time='00:00'):
	"""
	Schedule image files deletion from specific folder path every day at a specific time

    :param folder_path: path to specific folder
	:param scheduled_time: scheduled time (str)

	"""
	schedule.every().day.at(scheduled_time).do(delete_image_files)
	while True:
		schedule.run_pending()
		time.sleep(1)


def delete_image_files(folder_path=UPLOAD_FOLDER+'*'):
    """
    
    Delete all files from specific folder path

    :param folder_path: path to specific folder
    
    """
    for image_file in glob(folder_path):
      os.remove(image_file)


def run_inference(input_tensor, interpreter=MODEL):
    """
    Run inference with TF Lite model

    :param input_tensor: image file as Tensor
    :param interpreter: a loaded TF Lite model

    :return: predicted results (numpy array)

    """
    # Load TFLite model and allocate tensors.
    interpreter.allocate_tensors()
    # get input and output tensors
    input_details = interpreter.get_input_details()
    # set the tensor to point to the input data to be inferred
    input_index = interpreter.get_input_details()[0]["index"]
    # input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32)
    interpreter.set_tensor(input_index, input_tensor)
    # Run the inference
    interpreter.invoke()
    output_details = interpreter.get_output_details()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)
    return results


def detect_breed_from_img(img_path, img_size, model, dependency, object_detector=None):
    """
    
    Wrapper which use a trained machine learning model in order to detect a dog breed from a dog image

    :param img_path: path to image file
    :param img_size: image size (width, height)
    :param model: a TensorFlow classifier
    :param dependency: external dependencies (breeds labels)
    :param object_detector: object detection model (optional)

    :return: a dog breed and model accuracy

    """
    # Image preprocessing
    if object_detector is not None:
      img_as_array = object_detector.detect_objects(img_path,                                                   # Crop image
                                                    classes_kept="dog",
                                                    crop=True,
                                                    display=False,
                                                    verbose=False,
                                                    return_image=True)                             
    else:
      pil_img = tf.keras.preprocessing.image.load_img(img_path)
      img_as_array = tf.keras.preprocessing.image.img_to_array(pil_img)                                         # Load image file as Tensor

    img_as_array = tf.expand_dims(tf.image.resize(img_as_array, img_size), axis=0)                              # Resize & reshape
    img_as_array = tf.keras.applications.nasnet.preprocess_input(tf.cast(img_as_array, tf.float32))             # Normalize
    # Predict dog breed with trained model
    start_time = time.time()
    breed_predictions = model.predict(img_as_array)[0] # tf lite version -> breed_predictions = run_inference(img_as_array)
    end_time = time.time()
    print("Inference time : {}s".format(round(end_time-start_time, 2)))
    # tf lite version -> predicted_breed_idx = np.argmax(breed_predictions) 
    predicted_breed_idx = tf.argmax(breed_predictions).numpy()                                                  # predicted breed index
    predicted_breed_prob = round(breed_predictions[predicted_breed_idx]*100, 1)                                 # predicted breed accuracy
    predicted_breed_label = [breed for breed, idx in dependency.items() if idx == predicted_breed_idx][0]       # predicted breed label
    predicted_breed_label = re.sub('[^A-Za-z0-9]+', ' ', predicted_breed_label).capitalize()                    # Clean & capitalize breed label
    return predicted_breed_label, predicted_breed_prob


def get_grad_cam_img(img_path, img_size, model):
    """
    
    Wrapper which get Grad CAM version of a dog image
    
    :param img_path: path to image file
    :param img_size: image size (width, height)
    :param model: a TensorFlow classifier

    :return: a PIL image
    
    """
    # Preprocess original image based on pretrained model preprocessing input function
    img_array = tf.keras.applications.nasnet.preprocess_input(get_img_array(img_path, size=img_size))
    # Get pretrained model with Transfert Learning
    pretrained_model = model.layers[0]
    # Get last convolutional layer output layer
    last_layer_name = model.layers[0].layers[-1].name
    # Get Grad CAM image
    grad_cam_img = grad_cam(img_path, img_array, model, last_layer_name, pretrained_model)
    return grad_cam_img


def get_dog_breed_detector_results(image_file,
                                   upload_folder=UPLOAD_FOLDER,
                                   image_size=IMG_SIZE,
                                   model=MODEL,
                                   breeds=BREEDS,
                                   save_original_image_file=True):
    """
    
    Wrapper which get main data from Dog Breed Detector application
    It retrieves :
        - predicted breed, accuracy, 
        - Grad-CAM image (PIL format)
        - original image file location
    
    :param image_file: requested image file from flask request
    :param upload_folder: relative path to uploaded images
    :param img_size: image size (width, height)
    :param model: a TensorFlow classifier
    :param breeds: breeds labels
    :param save_original_image_file: enable/disable saving original image file locally (boolean)

    :return: a results dictionary

    """
    # Image processing
    image_location = os.path.join(upload_folder, image_file.filename)
    if save_original_image_file:
        image_file.save(image_location)
    # Predict dog breed 
    breed, accuracy = detect_breed_from_img(image_location, image_size, model, breeds) 
    # Get Grad CAM image
    grad_cam_img = get_grad_cam_img(image_location, image_size, model)
    # Resume main results
    results = {'breed': breed,
                'accuracy': accuracy,
                'grad-cam': grad_cam_img,
                'original-img-path': image_location
               }
    return results
