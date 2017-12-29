# tf_object_detector

## Motivation
This repo illustrates how one may use Tensorflow's Object Detector to detect various objects (e.g. person, dog, kite) within an image (including identifying where they are within the image).

## Environment

Instructions are based on running in an Ubuntu 16.04 LTS environment with Python 3.5 and based on the state of libraries and installation at the time of this writing.

## Setup
1. Ensure virtualenv and mkvirtualenvwrapper are installed.  Their installation instructions can be found here:
http://exponential.io/blog/2015/02/10/install-virtualenv-and-virtualenvwrapper-on-ubuntu/

2. Now create a virtual environment for this project:

    `mkvirtualenv --clear -p python3.5 tf-object-detector`
3. Clone the repo
3. Ensure CUDA 8 and CUDNN6 are installed. See instructions at https://gist.github.com/mjdietzx/0ff77af5ae60622ce6ed8c4d9b419f45
4. Install the requirements: `pip install -r requirements.txt`.  Note depending on your machine, you may wish to install Tensorflow manually.
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
8. Update PYTHONPATH to include path to models
`cd ..`
export PYTHONPATH=$PYTHONPATH:\`pwd\`:\`pwd\`/slim
9. Launch Jupyter notebook
`cd research/object_detection`
10. Launch `object_detection_tutorial.ipynb`
11. Cell > Run All to run demo, which downloads and uses a pre-trained model and applies to test images showing in boxes what was detected where with what accuracy.
