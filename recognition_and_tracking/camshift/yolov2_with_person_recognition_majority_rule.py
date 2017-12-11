#python libraries
import cv2
import numpy as np
import argparse
import os

#chainer
from chainer import serializers, Variable
import chainer.functions as F

#python scripts
from yolov2 import *
from CocoPredictor import *
from PersonClassifier import *
from network_structure import *


#preparing to save the video
def initWriter(w, h, fps, save_path):
    fourcc = cv2.VideoWriter_fourcc('F','L','V','1')
    rec = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
    return rec


#main
if __name__ == "__main__":

    # set some arguments
    parser = argparse.ArgumentParser(description='yolov2_darknet_predict for video')
    parser.add_argument('--video_file', '-v', type=str, default=False,help='path to video')
    parser.add_argument('--camera_ID', '-c', type=int, default=0,help='camera ID')
    parser.add_argument('--save_name', '-s', type=str, default=False,help='camera ID')
    args = parser.parse_args()

    # load a video file or connect to the web camera
    if not args.video_file == False:
        cap = cv2.VideoCapture(args.video_file)
    else:
        cap = cv2.VideoCapture(args.camera_ID)

    # set the writer for saving a result video
    if not args.save_name == False:

        save_path = 'results/videos/'

        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        ret, frame = cap.read()
        height, width, channels = frame.shape
        rec = initWriter(width, height, 30, save_path+args.save_name)

    # load YOLOv2
    coco_predictor = CocoPredictor()

    # load module type CNN1
    model_path1 = 'models/person_classifier/thibault_model5/'
    model1 = CNN_thibault2()
    image_size1 = 50
    person_classifier1 = PersonClassifier(model_path1,model1,image_size1)

    # load module type CNN2
    model_path2 = 'models/person_classifier/thibault_model_ultimate/'
    model2 = CNN_thibault_ultimate()
    image_size2 = 80
    person_classifier2 = PersonClassifier(model_path2,model2,image_size2)

    # recognition threshhold
    th = 0.9

    # prepare a window to show the recognition result
    cv2.namedWindow("video", cv2.WINDOW_NORMAL)

    # main loop
    while(True):

        # get a frame from the video/camera
        ret, frame = cap.read()

        if ret is not True:
            break

        # get the result of YOLOv2
        YOLO_results = coco_predictor(frame)

        # extruct information from result of YOLOv2
        for result in YOLO_results:

            left, top = result["box"].int_left_top()
            right, bottom = result["box"].int_right_bottom()
            color = (255, 0, 255)

            # Person : result["class_id"] == 0
            if result["class_id"] == 0:

                # input the person region to person_classifiers
                person_class1, prob1 = person_classifier1(frame[top:bottom, left:right])
                person_class2, prob2 = person_classifier2(frame[top:bottom, left:right])

                # calculate the mean value of 2 probabilities
                prob = (prob1 + prob2)*0.5

                #  condition for determination as "TARAGET"
                if person_class1 == 1 and person_class2 == 1 and prob > th:

                    color = (0,255,0)
                    result_info = 'TARGET(%2d%%)' % (prob*100)
                else:
                    result_info = 'OTHERS(%2d%%)' % ((1.0-prob)*100)

                print(result_info)

                # draw the result to the frame
                cv2.putText(frame, result_info, (left, bottom+25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # show the result
        cv2.imshow("video", frame)

        # write the frame to the video
        if not args.save_name == False:
            rec.write(frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

    cap.release()

    if not args.save_name == False:
        rec.release()
