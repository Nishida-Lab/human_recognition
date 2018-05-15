# human_recognition

Human recognition project for Tsukuba Challenge.
- record videos by multi cameras
- extract images from the video
- create overlayed image dataset
- train neural networks with chainer
- load trained chainer model and search humans

## Environment

* Python 2.7.6
* OpenCV (CV2) 2.4.8
* numpy
* matplotlib
* chainer

## Usage

1. Create a new directory and `git clone` this repository.
```
$ mkdir cnn_image_recognition/dataset && cd cnn_image_recognition
$ git clone https://github.com/Nishida-Lab/human_recognition.git
```
2. Download a sample dataset from following URL.  
https://drive.google.com/open?id=1YgOChz93w54ItvIDdwqjYVwV7zI4PFQe

3. Move each directories to the directory `dataset` according to the directory structure.

## Directory Structure

```
cnn_image_recognition       <- the directory you created
    ├── dataset             <- dataset
    │      ├── humans
    │      ├── backgrounds
    │      └── overlapped
    └── human_recognition   <- this repository
```

## Dataset preparation
### Overlap a human image on the background image
```
$ cd dataset_processing
$ python overlap.py -bp ../../dataset/backgrounds -hp ../../dataset/humans -sp ../../dataset/overlapped
```
```
Left click: change the position of the human
key P : expand the human
key N: shrink the human
key W: make the human wide
key H: make the human tall
key F: flip the human
key L: rotate the human counterclockwise
key R: rotate the human clockwise
key G: gamma correction (high brightness)
key D: gamma correction (low brightness)
key Z: gamma correction (high contrast)
key X: gamma correction (low contrast)
key A: smoothing (average filter)
key O: reset the human
key S: save the overlapped image
key Q: load the next images
key C: confirm current number of the images
key Esc: close everything
```
### Show dataset examples
```
$ cd dataset_processing
$ python show_dataset.py -dp ../../dataset/overlapped
```
### Dataset formats
labels.txt
```
1 (orange)
1 (orange)
2 (blue)
2 (blue)
1 (orange)
...
```
rectangles.txt
```
(center_x),(center_y),(width),(height)
1020,372,56,124
260,372,56,124
696,432,160,316
584,432,160,316
1032,392,96,224
248,392,96,224
848,432,300,332
432,432,300,332
1096,272,140,236
184,272,140,236
...
```
