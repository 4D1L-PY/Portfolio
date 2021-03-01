import re
import time
import cv2
import numpy as np
from PIL import Image
from glob import glob
from dog_breed_detector_package.config import tf, IMG_SIZE, BREEDS, MODEL



def get_img_array(img_path, size):
    """
    Transform image file (from file path) into image array

    :param img_path: image file path (absolute or relative)
    :param size: image size (width, height)
    
    :return: image as array

    """
    # 'img' is a PIL image of size w x h
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (w, h, n_ch)
    array = tf.keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, w, h, n_ch)
    array = np.expand_dims(array, axis=0)
    return array


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
    breed_predictions = model.predict(img_as_array)[0]
    end_time = time.time()
    print("Inference time : {}s".format(round(end_time-start_time, 2)))
    predicted_breed_idx = tf.argmax(breed_predictions).numpy()                                                  # predicted breed index
    predicted_breed_prob = round(breed_predictions[predicted_breed_idx]*100, 1)                                 # predicted breed accuracy
    predicted_breed_label = [breed for breed, idx in dependency.items() if idx == predicted_breed_idx][0]       # predicted breed label
    predicted_breed_label = re.sub('[^A-Za-z0-9]+', ' ', predicted_breed_label).capitalize()                    # Clean & capitalize breed label
    return predicted_breed_label, predicted_breed_prob


# Grad CAM
# Paper : https://arxiv.org/pdf/1610.02391.pdf
# Article : https://medium.com/@mohamedchetoui/grad-cam-gradient-weighted-class-activation-mapping-ffd72742243a


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, base_model=None):
    """

    Build Grad-CAM heatmap from image array and model data

    :param img_array: an image numpy array
    :param model: a TensorFlow classifier
    :param last_conv_layer_name: last convolutional output layer name
    :param base_model: base model from custom Sequential model (optional)
    
    :return: a heatmap (array)
    
    """
    # First, we create a model that maps the input image to the activations of the last conv layer
    if base_model is None:
        base_model = model
        last_conv_layer_idx = list(map(lambda layer: layer.name, base_model.layers)).index(last_conv_layer_name)
        classifier_layer_names = [layer.name for i, layer in enumerate(base_model.layers) if i > last_conv_layer_idx]
    else:
        classifier_layer_names = [layer.name for layer in model.layers if layer.name != base_model.name]
    last_conv_layer = base_model.get_layer(last_conv_layer_name)
    last_conv_layer_model = tf.keras.Model(base_model.inputs, last_conv_layer.output)
    # Second, we create a model that maps the activations of the last conv layer to the final class predictions
    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = tf.keras.Model(classifier_input, x)
    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        # Compute activations of the last conv layer and make the tape watch it
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        # Compute class predictions
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]
    # This is the gradient of the top predicted class with regard to the output feature map of the last conv layer
    grads = tape.gradient(top_class_channel, last_conv_layer_output)
    # This is a vector where each entry is the mean intensity of the gradient over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]
    # The channel-wise mean of the resulting feature map is our heatmap of class activation
    heatmap = np.mean(last_conv_layer_output, axis=-1)
    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap


def grad_cam(img_path, *args):
    """
    Generate Grad-CAM PIL image
    
    :param img_path: relative path to image file
    :param *args: make_gradcam_heatmap parameters

    :return: a PIL image

    """
    # Generate class activation heatmap
    heatmap = make_gradcam_heatmap(*args)
    # We load the original image
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    # We rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)
    # We use jet colormap to colorize heatmap
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    # Superimpose the heatmap on original image
    superimposed_img = np.float32(heatmap) + np.float32(img)
    superimposed_img = 255 * superimposed_img / np.max(superimposed_img)
    superimposed_img = superimposed_img[:, :, ::-1]
    superimposed_img = Image.fromarray(np.uint8(superimposed_img))
    return superimposed_img


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


def get_dog_breed_detector_results(img_path,
                                   image_size=IMG_SIZE,
                                   model=MODEL,
                                   breeds=BREEDS):
    """
    
    Wrapper which get main data from Dog Breed Detector application
    It retrieves :
        - predicted breed, accuracy, 
        - Grad-CAM image (PIL format)
        - original image file location
    
    :param img_path: requested image file from flask request
    :param img_size: image size (width, height)
    :param model: a TensorFlow classifier
    :param breeds: breeds labels

    :return: a results dictionary

    """
    # Predict dog breed
    breed, accuracy = detect_breed_from_img(img_path, image_size, model, breeds)
    # Get Grad CAM image
    grad_cam_img = get_grad_cam_img(img_path, image_size, model)
    # Resume main results
    results = {'breed': breed,
               'accuracy': accuracy,
               'grad-cam': grad_cam_img
              }
    return results
