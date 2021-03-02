# OpenClassrooms Machine Learning Engineer Path

## Project 6 : Classify Images Using Deep Learning Algorithms

### I - Project description :

This project is an API built with a pre-trained image classifier (NASNet) which detect a dog breed according to the dog present in an image.

**Problematic :**

_You are a volunteer for the animal welfare association in your neighborhood._

_You learn, by talking to a volunteer, that their database of residents begins to grow and that they do not always have time to reference the images of the animals that they have accumulated for several years._

_They would therefore like to obtain an algorithm capable of classifying the images according to the breed of the dog present in the image._

**Methodology :**

Training data was preprocessed by **cropping images** with **object detection algorithms** ([**Faster R-CNN + Inception Resnet V2**](https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1), [**MobileNet V2**](https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1)), in order to remove irrelevant objects (humans, other animals, etc ...) by targeting dogs specifically in the images.

Two learning methodologies were compared in order to pursue this project :

- **Training model from scratch** : build an image classifier from scratch based on CNN architecture
  - _N.B : sampling target is required due to few training data available_<br>
- **Training model with transfert learning** : build an image classifier from pretrained model and fine-tune model weights with transfert learning

Two hyperparameters optimization methods were also benchmarked :
- **evolutionary algorithm** implemented from scratch with [DEAP](https://deap.readthedocs.io/en/master/)
- **bayesian optimization** made easy with [optuna](https://optuna.org/)

**Evaluation results :**

|  Metric\Model   | Custom CNN * | NASNetMobile |  NASNetLarge |
| --------------  | --- | ----- | ----- |
|    Accuracy     | 50.7 |  85.5 |  **94.6** |

_* trained with a sample of 10 breeds (10% of total breeds)_

**Main data :**

The volunteers of the association did not have time to collect the various images of the residents scattered on their hard drives.

Algorithms were trained using the [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/).


**Project architecture :**

- **_applications_** folder which contains deployed Flask API, Android and Tkinter applications.
- **_notebooks_** folder which contains jupyter notebooks in .ipynb and HTML formats.
- **_src_** folder which contains several personal packages and modules coded in python.

**Project notebook :**

The HTML version of the modeling notebook was uploaded on an Amazon S3 bucket, and is available at the following link :
- **[P6_Dog_Breed_Image_Classification](https://ao-oc-mle-p6-dog-breed-detector.s3.eu-west-3.amazonaws.com/P6_Dog_Breed_Image_Classification.html)**


### II - Applications :

This project provides three applications which detect the breed of a dog from an image file.

- **[Dog Breed Detector Flask API](https://github.com/4D1L-PY/Portfolio/tree/main/OC-MLE/Dog-Breed-Detector/applications/flask_dog_breed_detector)** : API built with [Flask](https://flask.palletsprojects.com/en/1.1.x/), which detects a dog breed by uploading a dog image
- **[Dog Breed Detector Android App](https://github.com/4D1L-PY/Portfolio/tree/main/OC-MLE/Dog-Breed-Detector/applications/kivy_dog_breed_detector)** : Android application built with [Kivy](https://kivy.org/#home) and [KivyMD](https://kivymd.readthedocs.io/en/latest/) that allows users to detect a dog breed by uploading photos of dogs  
- **[Dog Breed Detector Tkinter program](https://github.com/4D1L-PY/Portfolio/tree/main/OC-MLE/Dog-Breed-Detector/applications/tkinter_dog_breed_detector)** : [Tkinter](https://docs.python.org/fr/3/library/tkinter.html) version of Dog Breed Detector Flask API

#### Try it

The Flask API was hosted with Google Cloud Platform App Engine service.

Click on the following link to try this API :
[Dog Breed Detector API](https://dog-breed-detector-306216.ew.r.appspot.com/)
