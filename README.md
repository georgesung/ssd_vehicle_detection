# Vehicle Detection with SSD in TensorFlow
This is adapted from [https://github.com/georgesung/ssd_tensorflow_traffic_sign_detection]

The code was slightly modified, and the model was trained using the Udacity vehicle detection dataset: [https://github.com/udacity/self-driving-car/tree/master/annotations]

Here is a demo video of video detection with this implementation: [https://www.youtube.com/watch?v=Ha1QbnuDYJU]

## Dependencies
* Python 3.5+
* TensorFlow v0.12.0
* Pickle
* OpenCV-Python
* Matplotlib (optional)

## How to run
*Note this was copy and pasted from the traffic sign detection project, the process is included here for reference. BUT, the link to the pre-trained model is updated :)*

Clone this repository somewhere, let's refer to it as `$ROOT`

To run predictions using the pre-trained model:
* [Download the pre-trained model](https://drive.google.com/open?id=0BzaCOTL9zhUldDVCcmRHVHlHRXc) to `$ROOT`
* `cd $ROOT`
* `python inference.py -m demo`
  * This will take the images from sample_images, annotate them, and display them on screen
* To run predictions on your own images and/or videos, use the `-i` flag in inference.py (see the code for more details)
  * Note the model severly overfits at this time

Training the model from scratch:
*TODO: Modify the steps below to work with vehicle dataset*
* Download the [LISA Traffic Sign Dataset](http://cvrr.ucsd.edu/LISA/lisa-traffic-sign-dataset.html), and store it in a directory `$LISA_DATA`
* `cd $LISA_DATA`
* Follow instructions in the LISA Traffic Sign Dataset to create 'mergedAnnotations.csv' such that only stop signs and pedestrian crossing signs are shown
* `cp $ROOT/data_gathering/create_pickle.py $LISA_DATA`
* `python create_pickle.py`
* `cd $ROOT`
* `ln -s $LISA_DATA/resized_images_* .`
* `ln -s $LISA_DATA/data_raw_*.p .`
* `python data_prep.py`
  * This performs box matching between ground-truth boxes and default boxes, and packages the data into a format used later in the pipeline
* `python train.py`
  * This trains the SSD model
* `python inference.py -m demo`
