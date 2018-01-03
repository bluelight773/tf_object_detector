# coding: utf-8
"""
Script/module illustrating use of TensorFlow's Object Detector API with a pre-trained model to detect various objects
in images.

When run as main, the detection will be applied to 2 test_images in models/research/object_detection.

Ensure PATH_TO_TF_MODELS_OBJECT_DETECTION points to the full path of models/research/object_detection
By default that path is assumed to be inside the repo root.

Note that the code was tested on an environment consisting of Ubuntu 16.04 LTS, Python 3.5 and TensorFlow 1.4.1.

The code is mainly based on:
https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
"""

import numpy as np
import os
import urllib.request
import sys
import tarfile
import tensorflow as tf

from matplotlib import pyplot as plt
from PIL import Image

# This is needed so that we ensure plots actually show by using pylab.show(). Note that they'll show in a blocking
# manner.
import pylab


if tf.__version__ < "1.4.0":
    raise ImportError("Please upgrade your tensorflow installation to v1.4.* or later!")

# Env setup

# Programmatically adding necessary paths to PYTHONPATH
PATH_TO_REPO_ROOT = os.path.dirname(os.path.realpath(__file__))
PATH_TO_TF_MODELS_OBJECT_DETECTION = os.path.join(PATH_TO_REPO_ROOT, "models", "research", "object_detection")
sys.path.append(os.path.join(PATH_TO_REPO_ROOT, "models", "research"))
sys.path.append(PATH_TO_TF_MODELS_OBJECT_DETECTION)

# Object detection imports
# Here are the imports from the object detection module.
from utils import label_map_util
from utils import visualization_utils as vis_util

# Model preparation

# Variables
# Any model exported using the 'export_inference_graph.py' tool can be loaded here simply by changing PATH_TO_CKPT to
# point to a new .pb file.
# 
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo]
# (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list
# of other models that can be run out-of-the-box with varying speeds and accuracies.

# What model to download.
MODEL_NAME = "ssd_mobilenet_v1_coco_2017_11_17"
MODEL_FILE = MODEL_NAME + ".tar.gz"
DOWNLOAD_BASE = "http://download.tensorflow.org/models/object_detection/"

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = os.path.join(PATH_TO_TF_MODELS_OBJECT_DETECTION, MODEL_NAME, "frozen_inference_graph.pb")

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(PATH_TO_TF_MODELS_OBJECT_DETECTION, "data", "mscoco_label_map.pbtxt")

NUM_CLASSES = 90

# Download Model if PATH_TO_CKPT doesn't already exist
if not os.path.exists(PATH_TO_CKPT):
    urllib.request.urlretrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if "frozen_inference_graph.pb" in file_name:
            tar_file.extract(file, PATH_TO_TF_MODELS_OBJECT_DETECTION)
    # Delete the downloaded tar having already extracted its contents
    os.remove(MODEL_FILE)

# Load a (frozen) TensorFlow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, "rb") as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name="")

# Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this
# corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping
# integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# Helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


# Detection

def show_detected_objects_in_images(image_paths, detection_graph=detection_graph):
    """Given a list of paths to images, display each image (one by one) with boxes and labels for detected objects."""
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name("image_tensor:0")
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name("detection_boxes:0")
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name("detection_scores:0")
            detection_classes = detection_graph.get_tensor_by_name("detection_classes:0")
            num_detections = detection_graph.get_tensor_by_name("num_detections:0")
            for image_path in image_paths:
                image = Image.open(image_path)
                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                image_np = load_image_into_numpy_array(image)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)
                # Use an output image size of 12 inches by 8 inches
                plt.figure(figsize=(12, 8))
                plt.imshow(image_np)
                pylab.show()


if __name__ == "__main__":
    # For the sake of simplicity we will use only the 2 images in test_images
    # image1.jpg
    # image2.jpg
    PATH_TO_TEST_IMAGES_DIR = os.path.join(PATH_TO_TF_MODELS_OBJECT_DETECTION, "test_images")
    TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, "image{}.jpg".format(i)) for i in range(1, 3)]
    show_detected_objects_in_images(TEST_IMAGE_PATHS)