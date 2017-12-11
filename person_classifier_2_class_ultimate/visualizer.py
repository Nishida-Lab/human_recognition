#python library
import numpy as np
from matplotlib import pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import json

#openCV
import cv2


#visualize loss reduction
def loss_visualizer(path):

    epoch = []
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []

    f = open(path+'/result/log', 'r') #load log file
    data = json.load(f)
    f.close()

    value = []

    for i in range(0,len(data)):
        value = data[i]
        epoch.append(value["epoch"])
        train_loss.append(value["main/loss"])
        test_loss.append(value["validation/main/loss"])
        train_accuracy.append(value["main/accuracy"])
        test_accuracy.append(value["validation/main/accuracy"])

    # accuracy graph
    plt.figure(10)
    plt.plot(epoch,train_accuracy,"b",lw=5,label = "train accuracy")
    plt.plot(epoch,test_accuracy,"g",lw=5,label = "validation accuracy ")
    plt.legend(fontsize=24,loc = "lower right")
    plt.xlabel("epoch",fontsize=30)
    plt.ylabel("accuracy",fontsize=30)
    plt.tick_params(labelsize=20)
    plt.tight_layout()

    # LOSS graph
    plt.figure(20)
    plt.plot(epoch,train_loss,"b",lw=5,label = "train LOSS")
    plt.plot(epoch,test_loss,"g",lw=5,label = "validation LOSS")
    #plt.yscale('log')
    plt.legend(fontsize=22)
    plt.xlabel("epoch",fontsize=30)
    plt.ylabel("LOSS",fontsize=30)
    plt.tick_params(labelsize=20)
    plt.tight_layout()


#visualize classification result
def result_visualizer(input_img, estimated_label):

    end_flag = False
    fontType = cv2.FONT_HERSHEY_SIMPLEX

    height = input_img.shape[0]
    width = input_img.shape[1]

    input_img = cv2.cvtColor(input_img,cv2.COLOR_RGB2BGR)

    if estimated_label == 1:
        print 'TARGET'
        cv2.putText(input_img, 'TARGET', (5, height-20),
                    fontType, 0.8, (88,217,78), 2, cv2.CV_AA)

    cv2.imshow('image', input_img)

    while (True):

        key = cv2.waitKey(1) & 0xFF

        # load next image
        if key & 0xFF == ord("z"):
            break

        # finish this program
        if key & 0xFF == 27:
            end_flag = True
            break

    return end_flag
