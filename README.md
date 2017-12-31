# tf_object_detector

## Motivation
This repo illustrates how one may use Tensorflow's Object Detector to detect various objects (e.g. person, dog, kite)
within an image or webcam video stream (including identifying where they are within the image/frame).

## Environment

Instructions are based on running in an Ubuntu 16.04 LTS environment with Python 3.5 and based on the state of libraries
and installation at the time of this writing including TensorFlow 1.4.1.

## Setup
1. Ensure virtualenv and mkvirtualenvwrapper are installed.  Their installation instructions can be found here:
http://exponential.io/blog/2015/02/10/install-virtualenv-and-virtualenvwrapper-on-ubuntu/

2. Now create a virtual environment for this project:

    `mkvirtualenv --clear -p python3.5 tf-object-detector`
3. Clone the repo
3. Ensure CUDA 8 and CUDNN6 are installed. See instructions at
https://gist.github.com/mjdietzx/0ff77af5ae60622ce6ed8c4d9b419f45
4. Install the requirements: `pip install -r requirements.txt`.  Note depending on your machine, you may wish to install
Tensorflow manually.
5. Clone tensorflow models directory: 

    `git clone github.com/tensorflow/models.git`
6. Install Protobuf Compiler. You may want to ensure that you're installing the latest version to avoid issues later on.

    `wget https://github.com/google/protobuf/releases/download/v3.5.1/protoc-3.5.1-linux-x86_64.zip`
    
    `unzip protoc-3.5.1-linux-x86_64.zip -d protoc3`
    
    `sudo mv protoc3/bin/* /usr/bin/`
    
    `sudo mv protoc3/include/* /usr/include/`
    
7. Run protocol buffer compiler against the object detector
`cd models/research`
`protoc object_detection/protos/*.proto --python_out=.`

## Code Overview
* `detect_objects_in_images.py`: Standalone script/module illustrating use of TensorFlow's Object Detector API with a
pre-trained model to detect various objects in an image.  If run as main the detection will be applied to 2 test_images 
in `models/research/object_detection`. If imported, the main function of interest will be 
`show_detected_objects_in_video`.
* `detect_objects_in_video.py`: Standalone script/module illustrating use of TensorFlow's Object Detector API with a
pre-trained model to detect various objects appearing in the default webcam video stream.