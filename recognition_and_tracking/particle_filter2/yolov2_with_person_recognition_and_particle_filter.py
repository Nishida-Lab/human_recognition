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
from particle_filter import *


#preparing to save the video
def initWriter(w, h, fps, save_path):
    fourcc = cv2.VideoWriter_fourcc('F','L','V','1')
    rec = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
    return rec

# display person detection result of CNNs
def draw_result(frame, result_info, left, top, right, bottom, color):
    cv2.putText(frame, result_info, (left, bottom+25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

# display current state
def info(frame, message, color):
    cv2.putText(frame, message, (10,18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

#Main
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

    ret, frame = cap.read()
    height, width, channels = frame.shape

    rec = False
    if not args.save_name == False:
        save_path = 'results/videos/'
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        rec = initWriter(width, height, 30, save_path+args.save_name)

    coco_predictor = CocoPredictor()

    model_path = 'models/person_classifier/thibault_model5/'
    model = CNN_thibault2()
    image_size = 50
    person_classifier = PersonClassifier(model_path, model, image_size)

    th = 0.9
    crop_param_w = 3
    crop_param_h = 4

    target_counter = 0
    target_center = (0,0)
    past_target_center = (0,0)
    image_size = (frame.shape[0], frame.shape[1])

    pf = ParticleFilter(image_size)
    pf.initialize()
    run_pf = RunParticleFilter(pf)
    run_pf.clear()
    PF_flag = False

    CNN_message = 'Searching with YOLOv2...'
    PF_message = 'Tracking with YOLOv2 + Particle Filter...'
    CNN_message_color = (255, 0, 255)
    PF_message_color = (204, 153, 51)
    message = CNN_message
    message_color = CNN_message_color

    cv2.namedWindow("video", cv2.WINDOW_NORMAL)

    while(True):

        ret, frame = cap.read()

        if ret is not True:
            break

        nms_results = coco_predictor(frame)

        for result in nms_results:

            left, top = result["box"].int_left_top()
            right, bottom = result["box"].int_right_bottom()
            color = (255, 0, 255)

            if result["class_id"] != 0:
                message = CNN_message
                message_color = CNN_message_color
                continue

            person_class, prob = person_classifier(frame[top:bottom, left:right])

            # flag!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            print(PF_flag)

            if person_class != 1 or prob < th:
                target_counter = 0
                print("ooooooo")
                if PF_flag:
                    print("run PF 1")
                    frame, PF_flag = run_pf.RUN_PF(frame, past_target_center)
                    message = PF_message
                    message_color = PF_message_color
                    continue
                else:
                    print("wwwwwwwwwww")
                    result_info = 'OTHERS(%2d%%)' % ((1.0-prob)*100)
                    draw_result(frame, result_info, left, top, right, bottom, color)
                    message = CNN_message
                    message_color = CNN_message_color
                    continue

            w = right - left
            h = bottom - top

            past_target_center = target_center
            target_center = (left+int(w*0.5),top+int(h*0.5))

            dist = np.linalg.norm(np.asarray(past_target_center)-np.asarray(target_center))

            target_counter += 1

            # past_target_center??

            if (target_counter > 5 and dist < 50) or PF_flag:
                print("run PF 1")
                frame, PF_flag = run_pf.RUN_PF(frame, target_center)
                message = PF_message
                message_color = PF_message_color
                continue
            else:
                print("qqqqqqqqqqqqqqqqqq")
                color = (0,255,0)
                result_info = 'TARGET(%2d%%)' % (prob*100)
                cv2.circle(frame, target_center, 5, (0, 215, 253), -1)
                draw_result(frame, result_info, left, top, right, bottom, color)
                run_pf.clear()
                message = CNN_message
                message_color = CNN_message_color

        info(frame, message, message_color)
        cv2.imshow("video", frame)

        if not args.save_name == False:
            rec.write(frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            end_flag = True

    cap.release()

    if not args.save_name == False:
        rec.release()
