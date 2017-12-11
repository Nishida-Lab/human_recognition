#python library
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import *
import time

#openCV
import cv2

#chainer library
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer import serializers

#python script
import network_structure as nn
import load_dataset as ld
import visualizer as v


#validation using learned CNN
def validation(size,model,x,y):

    validation_label = []
    success_counter = 0
    mistake_counter = 0
    end_flag = False

    for i in range(len(x)):

        input_img = cv2.cvtColor(cv2.imread(x[i]),cv2.COLOR_BGR2RGB)
        img = cv2.resize(input_img, (size,size), interpolation = cv2.INTER_LINEAR)
        img = img.reshape((1,3,size,size)).astype(np.float32) / 255.

        start = time.time()
        validation_output = model.forward(chainer.Variable(img))
        validation_output = validation_output.data
        print
        #print validation_output
        #print F.softmax(validation_output).data
        prob = F.softmax(validation_output).data[0][1]
        label =  np.argmax(validation_output)
        elapsed_time = time.time() - start

        validation_label.append(label)

        if label == 1:
            color = (0,255,0)
            result_info = "TARGET({0:.2f}%)".format(prob*100)
        else:
            result_info = "OTHERS({0:.2f})".format((1.0-prob)*100)

        print result_info

        print str(i+1) + " actual: " + str(y[i]) \
              + " estimated: " + str(label) \
              + " time:{0:.5f}".format(elapsed_time) + " [sec]"

        #visualize result
        #end_flag = v.result_visualizer(input_img, label)

        if label == y[i]:
            success_counter += 1.0

        if label == 1 and y[i] == 0:
            mistake_counter += 1.0
            # end_flag = v.result_visualizer(input_img, label)

        if end_flag == True:
            break

    cv2.destroyAllWindows()

    success_rate = (success_counter/i)*100
    mistake_rate = (mistake_counter/i)*100

    print "test accuracy: " + str(round(success_rate,2))
    print "mistake rate: " + str(round(mistake_rate,2))

    return validation_label


#main
if __name__ == '__main__':

    test_N = 1000

    model_path = 'thibault_model5/'
    model_name = 'cnn_gpu.model'
    size = 50
    model = nn.CNN_thibault2()

    # model_path = 'thibault_model_ultimate/'
    # model_name = 'cnn_gpu.model'
    # size = 80
    # model = nn.CNN_thibault_ultimate()

    serializers.load_npz(model_path+model_name, model)
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    Xts, Yts = ld.load_test_dataset(test_N)
    print Xts.shape
    Ye = validation(size,model,Xts,Yts)

    v.loss_visualizer(model_path)
    plt.show()
