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
    end_flag = False

    th = 5

    for i in range(len(x)):

        input_img = cv2.cvtColor(cv2.imread(x[i]),cv2.COLOR_BGR2RGB)
        img = cv2.resize(input_img, (size,size), interpolation = cv2.INTER_LINEAR)
        img = img.reshape((1,3,size,size)).astype(np.float32) / 255.

        start = time.time()
        validation_output = model.forward(chainer.Variable(img))
        validation_output = validation_output.data

        print
        print validation_output
        label =  np.argmax(validation_output)


        if label==1 :
            if validation_output[0][label] < th:
                label = 0

        elapsed_time = time.time() - start

        validation_label.append(label)

        print str(i) + " actual: " + str(y[i]) \
              + " estimated: " + str(label) \
              + " time:{0:.5f}".format(elapsed_time) + " [sec]"

        #end_flag = v.result_visualizer(input_img, label)

        if label == y[i]:
            success_counter += 1.0

        if end_flag == True:
            break

    cv2.destroyAllWindows()

    success_rate = (success_counter/i)*100
    print "validation accuracy: " + str(round(success_rate,2))
    return validation_label


#main
if __name__ == '__main__':

    test_N = 100

    model_path = 'first_model/'
    model_name = 'cnn_gpu.model'
    size = 32

    model = nn.CNN_classifier2()

    serializers.load_npz(model_path+model_name, model)
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    Xts, Yts = ld.load_test_dataset(test_N)
    Ye = validation(size,model,Xts,Yts)

    # Xtr, Ytr, Xv, Yv = ld.load_dataset(100, 100)
    # Ye = validation(size,model,Xtr,Ytr)

    #v.loss_visualizer(model_path)
    #plt.show()
