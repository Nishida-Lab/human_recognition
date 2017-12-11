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


# save label list
def save_label_list(label_list,file_address_name):

    f = open(file_address_name,'w')
    for label in label_list:
        f.write(str(label) + '\n')


#main
if __name__ == "__main__":

    #video_path = '../../dataset/videos/gairan_1/camera_1.avi'
    #save_path = '../../dataset/images/overlaped/2017_08_18/for_training/humans/11/'

    parser = argparse.ArgumentParser(description='extract person images')
    parser.add_argument('--video_path', '-v', type=str, default=False,help='path to video')
    parser.add_argument('--save_no', '-s', type=int, default=False,help='save directory number')
    args = parser.parse_args()

    video_path = args.video_path
    save_path = '../../dataset/images/overlaped/2017_08_18/for_training/humans/'+str(args.save_no)+'/'

    cap = cv2.VideoCapture(video_path)

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    coco_predictor = CocoPredictor()

    frame_cnt = 0
    person_cnt = 0

    person_label = []

    cv2.namedWindow("video", cv2.WINDOW_NORMAL)

    while(True):

        ret, frame = cap.read()
        frame_cnt += 1

        if ret is not True:
            break

        nms_results = coco_predictor(frame)

        for result in nms_results:

            left, top = result["box"].int_left_top()

            right, bottom = result["box"].int_right_bottom()

            if result["class_id"] == 0:

                x1, y1 = result["box"].int_left_top()
                x2, y2 = result["box"].int_right_bottom()

                cv2.imwrite(save_path+str(person_cnt)+'.jpg', frame[y1:y2, x1:x2])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                text = '%s(%2d%%)' % (result["label"], result["probs"].max()*result["conf"]*100)
                print(text)
                cv2.putText(frame, text, (left, top-7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                person_label.append(0)
                person_cnt += 1

        cv2.imshow("video", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

    cap.release()

    #save_label_list(person_label, save_path+'labels.txt')
    print(frame_cnt)
    print(person_cnt)
    print(len(person_label))
