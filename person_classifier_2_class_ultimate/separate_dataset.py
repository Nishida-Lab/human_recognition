# python library
import os
import getpass
import matplotlib.pyplot as plt
import numpy as np
import imghdr # image file type checker
import random

# OpenCV
import cv2


# save list as .txt data
def save_list(label_list,file_address_name):

    f = open(file_address_name,'w')
    for label in label_list:
        f.write(str(label) + '\n')


# main
if __name__ == '__main__':

    train_validation_N = 520000
    test_N = 20000
    txt_path = 'data/'

    all_images_file = open(txt_path+'images.txt','r')
    all_labels_file = open(txt_path+'labels.txt','r')

    all_images = []
    all_labels = []

    for file_name in all_images_file:
        all_images.append(file_name.split('\n')[0])

    for label in all_labels_file:
        all_labels.append(int(label))

    print len(all_images)
    print len(all_labels)

    # print all_images[0:10]
    # print all_labels[0:10]

    data_N = len(all_images)

    random_index = range(data_N)
    random.shuffle(random_index)

    # print random_index[0:10]
    # print len(random_index)

    train_validation_x = []
    train_validation_y = []

    test_x = []
    test_y = []

    for i in random_index[0:train_validation_N]:
        train_validation_x.append(all_images[i])
        train_validation_y.append(all_labels[i])

    for i in random_index[train_validation_N:train_validation_N+test_N]:
        test_x.append(all_images[i])
        test_y.append(all_labels[i])

    print len(train_validation_x)
    print len(train_validation_y)
    print len(test_x)
    print len(test_y)

    # save training data
    save_list(train_validation_x, txt_path+'train_validation_x.txt')
    save_list(train_validation_y, txt_path+'train_validation_y.txt')

    # save test data
    save_list(test_x, txt_path+'test_x.txt')
    save_list(test_y, txt_path+'test_y.txt')
