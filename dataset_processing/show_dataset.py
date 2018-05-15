# -*- coding: utf-8 -*-

# Python libraries
import os
import numpy as np
from numpy.random import *
from matplotlib import pylab as plt
import random
import argparse

# OpenCV
import cv2


# load the dataset and display
def display_dataset(dataset_path):

    rec_file = open(dataset_path+'rectangles.txt','r')
    label_file = open(dataset_path+'labels.txt','r')

    rec_data = []
    label_data = []

    for rec_f in rec_file:
        rec_str = (rec_f.split(','))
        rec_data.append([int(rec_str[0]),int(rec_str[1]),int(rec_str[2]),int(rec_str[3])])

    for label_f in label_file:
        label_str = label_f
        label_data.append(int(label_str))

    index_list = random.sample(xrange(len(rec_data)), 3)

    for index in index_list:

        img = cv2.imread(dataset_path+str(index)+'.jpg')

        if label_data[index] == 1: rec_color = (0,165,255)
        elif label_data[index] == 2: rec_color = (255,125,86)
        else: rec_color = (0,0,0)

        cv2.rectangle(img, \
                      (rec_data[index][0] - rec_data[index][2]/2,  \
                       rec_data[index][1] - rec_data[index][3]/2), \
                      (rec_data[index][0] + rec_data[index][2]/2,  \
                       rec_data[index][1] + rec_data[index][3]/2), \
                      rec_color, 10)

        cv2.namedWindow('data'+str(index), cv2.WINDOW_NORMAL)
        cv2.imshow('data'+str(index),img)


# main
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='image dataset for training cnn')
    parser.add_argument('--dataset_path', '-dp', type=str, default="data",help='background image path')
    args = parser.parse_args()

    dataset_path = args.dataset_path + '/'

    display_dataset(dataset_path)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
