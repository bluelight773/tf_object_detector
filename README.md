# TensorFlow Object Detector

This repo is largely based on the following:
* [TensorFlow Object Detection Notebook](https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb)
* [Raccoon Detector Dataset and Tools](https://github.com/datitran/raccoon_dataset/)
* [PythonProgramming on TensorFlow Object Detection](https://pythonprogramming.net/introduction-use-tensorflow-object-detection-api-tutorial/)

## Motivation
This repo illustrates how one may use Tensorflow's Object Detector to detect various objects (e.g., person, dog, kite)
within an image or webcam video stream using pretrained models as well as how to build and use one's own model (via
finetuning).

## Environment

Instructions are based on running in an Ubuntu 18.04 LTS environment with Python 3.6.8 and based on the state of libraries
and installation at the time of this writing.  TensorFlow 1.13.1 was used.  Key library version are specified in
`requirements.txt`.

## Setup
1. Ensure venv is installed to use a virtual environment. The following command works for Python 3.6:

    `sudo apt-get install -y python3.6-venv`

2. Now create a virtual environment for this project using a command such as the following:

    `python3 -m venv ~/envs/tf_object_detector_env`
3. Clone the repo
4. Ensure CUDA 10.0 and CUDNN 7.5.0 are installed. Installation instructions for Ubuntu 18.04 can be found 
[here](https://medium.com/@zhanwenchen/install-cuda-10-1-and-cudnn-7-5-0-for-pytorch-on-ubuntu-18-04-lts-9b6124c44cc).
5. Activate the virtual environment with a command such as the following. Note that deactivation can be done using 
`deactivate`:

    `source ~/envs/tf_object_detector_env/bin/activate`
6. Ensure pip is updated:

    `pip install --upgrade pip`
7. Install the requirements: `pip install -r requirements.txt`.  Note depending on your machine, you may wish to install
Tensorflow manually.
8. Clone tensorflow models directory into the repo root: 

    `git clone https://github.com/tensorflow/models.git`
9. Install Protobuf Compiler. At the time of this writing, Protobuf Compiler version 3.5.1 was used, which works well
with the other library versions specified in `requirements.txt`.  If you use the latest library versions, then you'll
likely want to ensure you also use the latest Protobuf Compiler.
    ```
    wget https://github.com/google/protobuf/releases/download/v3.5.1/protoc-3.5.1-linux-x86_64.zip
    unzip protoc-3.5.1-linux-x86_64.zip -d protoc3
    sudo mv protoc3/bin/* /usr/bin/    
    sudo mv protoc3/include/* /usr/include/
    ```
    
10. Run protocol buffer compiler against the object detector
    ```
    cd models/research
    protoc object_detection/protos/*.proto --python_out=.
    ```

11. Ensure tkinter is installed for the purpose of displaying images when running a Python script from the command line
    `sudo apt-get install python3-tk`

## Using Pre-trained Object Detection Models
* `detect_objects_in_images.py`: Standalone script/module illustrating use of TensorFlow's Object Detector API with a
pre-trained model to detect various objects in an image.  If run as main, the detection will be applied to 2 test_images 
in `models/research/object_detection`. If imported, the main function of interest will be 
`show_detected_objects_in_images`.
* `detect_objects_in_video.py`: Standalone script/module illustrating use of TensorFlow's Object Detector API with a
pre-trained model to detect various objects appearing in the default webcam video stream. Press 'x' to exit.
* Assuming you'd followed the Setup instructions, key variables of interest if you'd prefer to use a different
pretrained model are: MODEL_NAME, PATH_TO_LABELS, and NUM_CLASSES.

## Building and Using your own Model (via Finetuning)
These steps describe how to build a model to detect a new label (ie, object) of interest.
Note that the step for generating TF records below at the moment doesn't support having multiple types
of objects (ie, multiple labels) although it may be quite simple to add such support.

1. Collect images from whatever sources you may have, Google, Bing and/or ImageNet for your object and place in a
"dataset" folder inside the repo root. Ensure many of the images show a lot of context (rather than simply having the
object of interest cover virtually the whole image).  Also, ideally aim for variety in the images, such as in terms of 
scale, pose and lighting while also ensuring a good portion may resemble what you expect the real-world input to look
like.  Also, preferably ensure image filenames don't have periods in them, or even better to restrict to alphanumeric
characters due to potential issues that may arise later when using labelImg tool.  The more images the better.  For good
performance you'll likely need 1000+ training images, but if you just want a quick illustrative run that'll likely work
in a few simple cases and fail in many others, then you may start out with around 100 training images.

2. Label images via labelimg (https://github.com/tzutalin/labelimg).
To install and use labelimg tool, while in the tf_object_dector repo root and in the tf_object_detector_env virtual
environment, run:
```
    git clone git@github.com:tzutalin/labelImg.git
    sudo apt-get install pyqt5-dev-tools
    cd labelImg
    make qt5py3
    pip install sip PyQt5
    python labelImg.py
```

3. Open the directory with your images and start labelling using "Create RectBox", and store the resulting XML per
image in the "dataset" folder.

4. Make a a "test" and "train" directory inside the "dataset" folder.  Copy 10% of images and their corresponding XMLs
into "test" and the remaining 90% into "train".   As of this writing, we need to have some kind of copy of the files
in both the "dataset" folder as well as in the "train" and "test" folder - at least due to `generate_tfrecord.py`
expecting to find files in the "dataset" folder.  You may prefer to rely on symlinks, such as to first have a symlink
for each file and then just split the symlinks between "train" and "test".  The following command allows one to create
symlinks in bulk, which you may want to use before creating the "test" and "train" folders.
`cp -as /full/path/to/dataset/* /full/path/to/dataset_symlinks"`

5. Generate a "train_labels.csv" and "test_labels.csv" corresponding to the XMLs:  Ensure the "dataset" folder is in the
working directory as should be the case if you followed the above instruction above and are in the repo root.  Then run
`python xml_to_csv.py` and you should have the CSVs inside the "dataset" folder.

6. Generate TF Records for the train and test data, so that the data is in a form amicable to TensorFlow: Update 
`generate_tfrecord.py` so that LABEL_NAME is the label name you used for your object of interest, then run:

    ```
    python generate_tfrecord.py --csv_input=dataset/train_labels.csv  --output_path=dataset/train.record
    python generate_tfrecord.py --csv_input=dataset/test_labels.csv  --output_path=dataset/test.record
    ```

7. Create `dataset/label_map.pbtxt` that maps an integer to each label name (whatever name you'd used earlier) starting
at 1, such as:
    ```
    item {
        id: 1
        name: 'label_name'
    }
    ``` 
8. Select and download the base pretrained model of interest. 

    A list of pretrained models can be found here:
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

    Select a model best suited for your purposes. For instance, if you need real-time speed, you may prefer to sacrifice
    precision for speed.  Also, you may wish to select a model that relied on a dataset with labels closest to your
    label of interest.

    Download the pretrained model of interest, such as:
    ```
    wget http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2017_11_17.tar.gz
    ```
    
    Extract the contents.  In this case, we'd have a "ssd_inception_v2_coco_2017_11_17" folder inside the repo root.


9. Select and download a corresponding base config file of interest.
    Corresponding configuration files may be found here:
https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs

    For instance, let's say we want to use the default pipeline configuration file for the pretrained inception v2 COCO
    model, then we may download the following config into the repo root.  Note that the link was obtained by clicking on
    "Raw" when viewing the file in github.
    
    ```
    wget https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/samples/configs/ssd_inception_v2_coco.config
    ```


10. Update the config file, so that it works for your dataset.

    Set `num_classes` to 1.
    
    Set `fine_tune_checkpoint` to point to the model of interest with a full path, such as 
`/full/path/to/ssd_inception_v2_coco_2017_11_17/model.ckpt`.

    Set `train_input_reader`' `input_path` to `/full/path/to/dataset/train.record`.

    Set `train_input_reader`'s `label_map_path` to `/full/path/to/dataset/label_map.pbtxt`.

    Set `eval_input_reader`' `input_path` to `/full/path/to/dataset/test.record`.

    Set `eval_input_reader`'s `label_map_path` to `/full/path/to/dataset/label_map.pbtxt`.
    
    Note you may lower the `batch_size` if you're running into memory error.
`

11. In the repo root, create an empty "tf_run" folder that contains empty "train" and "eval" folders for TensorFlow
output during training and evaluation.

12. Begin training your model (which will be the result of finetuning the pretraining model):
    ```
    cd models/research
    export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
    cd object_detection
    python train.py --logtostderr --train_dir=../../../tf_run/train --pipeline_config_path=../../../ssd_inception_v2_pets.config
    ```

13. Once training has begun, initiate the evaluation process (say in a different terminal window), which will poll the
training directory to obtain precision metrics against the test data.
    ```
    cd models/research
    export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
    cd object_detection
    python eval.py --logtostderr --eval_dir=../../../tf_run/eval --pipeline_config_path=../../../ssd_inception_v2_pets.config --checkpoint_dir=../../../tf_run/train
    ```
    If you run the training and evaluation simultaneously, you may end up with a GPU memory error.  One workaround is to
    add the following lines at the top of `eval.py`, so that the evaluation process uses the CPU instead of the GPU.
    ```
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    ```
    
    Alternatively, you could just rely on the training process, and can optionally run evaluation after training has
    been terminated following a given checkpoint.
    

14. While training and evaluation are running, you can track progress using tensorboard. From repo root, you can launch
tensorboard as follows then view the progress at localhost:6006 paying close attention to the TotalLoss (based on 
training data) and Mean Average Precision (based on test data).

    ```
    tensorboard --logdir=tf_run/
    ```

15. Terminate the training process once you believe good performance has been achieved.  Rough guidelines are to aim for
total loss (against the training data) to be under 1 and the mean average precision to be minimally above 80%.  Ideally,
you'd want to stop at a point when both metrics are improving but right before the total loss' decline starts outpacing
the average precision's improvement as that suggests overfitting.  As suggested earlier, these metrics can all be
tracked in tensorboard.  You may have to wait roughly several thousand steps, but the number of steps may vary greatly
depending on the data and parameters. 

16. Export the inference graph then use it to do the object detection. Update the commandline parameters as needed in
your scenario though only the ckpt-#### will need updating if you followed the instructions (including model, config and
path selection) as is above.  Note that you should use the model.ckpt-#### corresponding to the step of interest (e.g.
when the checkpoint with the lowest totalloss was achieved).
    ```
    cd models/research
    export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
    cd object_detection

    python export_inference_graph.py \
        --input_type image_tensor \
        --pipeline_config_path ../../../ssd_inception_v2_pets.config \
        --trained_checkpoint_prefix ../../../training/model.ckpt-#### \
        --output_directory ../../../output_inference_graph
    ```

17. Use the new model.  If you followed the same file/folder naming conventions in the instruction above, just had
1 class (label), and your images were in .jpg format, then you can now simply run 
`python detect_objects_in_images_by_custom_model.py` or  `python detect_objects_in_video_by_custom_model.py` to see your
model in action.

    In case you had more than 1 class and/or used different names for files or folders, then ensure you update the
    necessary variables before running the Python scripts:
    * Ensure `MODEL_NAME` points to the output directory used for the inference graph (e.g. output_inference_graph)
    * Ensure `PATH_TO_LABELS` points to label map file created earlier (e.g. "dataset/label_map.pbtxt").
    * Ensure `NUM_CLASSES` is set to the number of labels you'd used.
    * Ensure `PATH_TO_TEST_IMAGES_DIR` points to the test images folder (eg. "dataset/test") containing at least 3 .jpg
    images

