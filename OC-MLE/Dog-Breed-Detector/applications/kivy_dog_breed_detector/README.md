# Dog Breed Detector Android App

Dog Breed Detector KivyMD application packaged for Android which use Flask API in order to make breed predictions.

## Demo

![](img/demo/dog_breed_detector_android.gif)


## Application architecture

This python application was built with [buildozer](https://buildozer.readthedocs.io/en/latest/index.html) which enable to package mobiles application for Android or iOS.  

- **_bin_** folder which contains the compiled .apk file which can be installed on Android phone/plateform.
- **_img_** folder which contains all image files used by kivy application.
- **_buildozer.spec_** [Buildozer application configuration file](https://buildozer.readthedocs.io/en/latest/specifications.html)
- **_main.kv_** KivyMD widgets file used by main.py.
- **_main.py_** KivyMD application file.

In order to package this KivyMD application, follow these instructions :

- [Install buildozer](https://buildozer.readthedocs.io/en/latest/installation.html)
- Create virtual environment and install application modules/packages requirements
  ```
  python -m venv environment

  source environment/bin/activate

  pip install -r requirements.txt
  ```
- [Deploy application on Android or iOS](https://buildozer.readthedocs.io/en/latest/quickstart.html)
- Install the compiled application on Android/iOS device which is in bin folder
