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

    parser = argparse.ArgumentParser(description='yolov2_darknet_predict for video')
    parser.add_argument('--video_file', '-v', type=str, default=False,help='path to video')
    parser.add_argument('--camera_ID', '-c', type=int, default=0,help='camera ID')
    parser.add_argument('--save_name', '-s', type=str, default=False,help='camera ID')
    args = parser.parse_args()

    if not args.video_file == False:
        cap = cv2.VideoCapture(args.video_file)
    else:
        cap = cv2.VideoCapture(args.camera_ID)

    if not args.save_name == False:

        save_path = 'results/videos/'

        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        ret, frame = cap.read()
        height, width, channels = frame.shape
        rec = initWriter(width, height, 30, save_path+args.save_name)


    # model_path = 'models/person_classifier/thibault_model5/'
    # model = CNN_thibault2()
    # image_size = 50

    model_path = 'models/person_classifier/thibault_model_ultimate/'
    model = CNN_thibault_ultimate()
    image_size = 80

    coco_predictor = CocoPredictor()
    person_classifier = PersonClassifier(model_path,model,image_size)
    th = 0.9

    cv2.namedWindow("video", cv2.WINDOW_NORMAL)

    while(True):

        ret, frame = cap.read()

        if ret is not True:
            break

        nms_results = coco_predictor(frame)

        # draw result
        for result in nms_results:
            left, top = result["box"].int_left_top()
            right, bottom = result["box"].int_right_bottom()
            color = (255, 0, 255)

            if result["class_id"] == 0:
                person_class, prob = person_classifier(frame[top:bottom, left:right])

                if person_class == 1:
                    if prob > th:
                        color = (0,255,0)
                        result_info = 'TARGET(%2d%%)' % (prob*100)
                    else:
                        result_info = 'OTHERS(%2d%%)' % ((1.0-prob)*100)
                else:
                    result_info = 'OTHERS(%2d%%)' % ((1.0-prob)*100)

                print(result_info)

                cv2.putText(frame, result_info, (left, bottom+25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            #cv2.rectangle(frame, (left, top), (right, bottom), rec_color, 3)
            #text = '%s(%2d%%)' % (result["label"], result["probs"].max()*result["conf"]*100)
            #cv2.putText(frame, text, (left, top-7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            #print(text)

        cv2.imshow("video", frame)

        if not args.save_name == False:
            rec.write(frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

    cap.release()

    if not args.save_name == False:
        rec.release()
