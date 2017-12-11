# python library
import os
import getpass
import matplotlib.pyplot as plt
import numpy as np
import imghdr # image file type checker
import random

# OpenCV
import cv2


# load dataset
def load_data(txt_path,images_file_name, labels_file_name):

    images_file = open(txt_path+images_file_name,'r')
    labels_file = open(txt_path+labels_file_name,'r')

    images = []
    labels = []

    for image_path in images_file:
        images.append(image_path.split('\n')[0])

    for label in labels_file:
        labels.append(int(label))

    return images, labels


# display dataset
def show_dataset(images, labels):

    end_flag = False
    fontType = cv2.FONT_HERSHEY_SIMPLEX
    size = 100

    for i in range(len(images)):

        print '\n' + images[i]

        img = cv2.imread(images[i])

        img = cv2.resize(img, (size,size), interpolation = cv2.INTER_LINEAR)

        height = img.shape[0]
        width = img.shape[1]

        if labels[i] == 1:
            print 'TARGET'
            cv2.putText(img, 'TARGET', (5, height-20), fontType, 0.8, (88,217,78), 2, cv2.CV_AA)

        else:
            print 'OTHERS'

        cv2.imshow('image ', img)

        while (True):

            key = cv2.waitKey(1) & 0xFF

            # load next image
            if key & 0xFF == ord("z"):
                break

            # finish this program
            if key & 0xFF == 27:
                end_flag = True
                break

        if end_flag == True:
            break

        cv2.destroyAllWindows()


# generate permuted array for img and label
def generate_random_array(images_array, labels_array):

    random_images_list = []
    random_labels_list = []

    data_N = len(images_array)
    random_index = np.random.permutation(data_N)

    for n in range(data_N):

        random_images_list.append(images_array[random_index[n]])
        random_labels_list.append(labels_array[random_index[n]])

    random_images_array = np.array(random_images_list)
    random_labels_array = np.array(random_labels_list).astype(np.int32)

    return random_images_array, random_labels_array


# load dataset for training
def load_dataset(train_N, validation_N):

    # 1. load txt file (image file names and labels)
    txt_path = 'data/'
    images_list, labels_list = load_data(txt_path,'train_validation_x.txt','train_validation_y.txt')

    images_array = np.array(images_list)
    labels_array = np.array(labels_list)

    print images_array.shape
    print labels_array.shape

    print 'total image N: ' + str(len(images_array))

    # 2. permutation
    random_images_array, random_labels_array = generate_random_array(images_array, labels_array)

    # 3. pick up dataset for training and validation
    train_images = random_images_array[0:train_N]
    train_labels = random_labels_array[0:train_N]

    validation_images = random_images_array[train_N:train_N+validation_N]
    validation_labels = random_labels_array[train_N:train_N+validation_N]

    return train_images, train_labels, validation_images, validation_labels


# load dataset for training
def load_test_dataset(test_N):

    # 1. load txt file (image file names and labels)
    txt_path = 'data/'
    images_list, labels_list = load_data(txt_path,'test_x.txt','test_y.txt')

    images_array = np.array(images_list)
    labels_array = np.array(labels_list)

    print images_array.shape
    print labels_array.shape

    print 'total image N: ' + str(len(images_array))

    # 2. permutation
    random_images_array, random_labels_array = generate_random_array(images_array, labels_array)

    # 3. pick up dataset for training and validation
    test_images = random_images_array[0:test_N]
    test_labels = random_labels_array[0:test_N]

    return test_images, test_labels


# main
if __name__ == '__main__':

    train_N = 170000
    validation_N = 20000
    test_N = 100

    Xtr, Ytr, Xv, Yv = load_dataset(train_N, validation_N)
    Xts, Yts = load_test_dataset(test_N)

    print type(Xtr[0])
    print type(Ytr[0])

    print Xtr.shape
    print Ytr.shape

    print Xv.shape
    print Yv.shape

    print Xts.shape
    print Yts.shape

    show_dataset(Xtr, Ytr)
    #show_dataset(Xts, Yts)
