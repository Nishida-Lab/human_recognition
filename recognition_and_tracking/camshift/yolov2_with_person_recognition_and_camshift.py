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
from camshift import *


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

    rec = False
    if not args.save_name == False:

        save_path = 'results/videos/'

        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        ret, frame = cap.read()
        height, width, channels = frame.shape
        rec = initWriter(width, height, 30, save_path+args.save_name)

    coco_predictor = CocoPredictor()

    model_path1 = 'models/person_classifier/thibault_model5/'
    model1 = CNN_thibault2()
    image_size1 = 50
    person_classifier = PersonClassifier(model_path1,model1,image_size1)

    th = 0.9

    camshift_scale = 0.1

    cv2.namedWindow("video", cv2.WINDOW_NORMAL)
    end_flag = False
    target_counter = 0
    target_center = (0,0)
    past_target_center = (0,0)

    while(True):

        ret, frame = cap.read()

        if ret is not True or end_flag is True:
            break

        nms_results = coco_predictor(frame)

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

                        w = right-left
                        h = bottom-top

                        past_target_center = target_center
                        target_center = (left+int(w*0.5),top+int((bottom-top)*0.5))

                        dist = np.linalg.norm(np.asarray(past_target_center)-np.asarray(target_center))

                        if dist < 50:
                            # tracking with camshift
                            end_flag = camshift(cap,rec, \
                                                target_center[0]-int(w*camshift_scale), \
                                                target_center[1]-2*int(h*camshift_scale), \
                                                int(w*camshift_scale), \
                                                int(h*camshift_scale))

                            # end_flag = camsift(cap,rec, left, top, width, height)

                        cv2.circle(frame, target_center, 2, (0, 215, 253), 2)

                    else:
                        result_info = 'OTHERS(%2d%%)' % ((1.0-prob)*100)
                else:
                    result_info = 'OTHERS(%2d%%)' % ((1.0-prob)*100)

                #print(result_info)

                cv2.putText(frame, result_info, (left, bottom+25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        cv2.putText(frame, 'Searching with YOLOv2...', (10,18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        cv2.imshow("video", frame)

        if not args.save_name == False:
            rec.write(frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            end_flag = True

    cap.release()

    if not args.save_name == False:
        rec.release()
