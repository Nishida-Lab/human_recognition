#python libraries
import cv2
import numpy as np
import argparse
import os
from collections import deque

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

# setup tracker
def setup_tracker():
    tracker = cv2.TrackerKCF_create() # KCF
    # tracker = cv2.TrackerMedianFlow_create() # MedianFlow
    return tracker


#Main
if __name__ == "__main__":

    # setup some arguments
    parser = argparse.ArgumentParser(description='yolov2_darknet_predict for video')
    parser.add_argument('--video_file', '-v', type=str, default=False,help='path to video')
    parser.add_argument('--camera_ID', '-c', type=int, default=0,help='camera ID')
    parser.add_argument('--save_name', '-s', type=str, default=False,help='camera ID')
    parser.add_argument('--label_file', '-l', type=str, default=False,help='path to label')
    args = parser.parse_args()

    # load teacher labels
    label_path = 'labels/' + args.label_file
    label_file = open(label_path,'r')
    label_list = []
    correct_label_counter = 0

    for label in label_file:
        l = int(label.split('\n')[0])
        label_list.append(l)
        if l == 1:
            correct_label_counter += 1

    label_N = len(label_list)
    print("labels: "+str(label_N))

    # prepare to get image frames
    if not args.video_file == False:
        cap = cv2.VideoCapture(args.video_file)
    else:
        cap = cv2.VideoCapture(args.camera_ID)

    ret, frame = cap.read()
    height, width, channels = frame.shape
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("input fps: " + str(fps))

    # prepare to record video
    rec = False
    if not args.save_name == False:
        save_path = 'results/videos/'
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        rec = initWriter(width, height, fps, save_path+args.save_name)

    # load YOLO model and weights
    coco_predictor = CocoPredictor()

    # load module type CNN model and weights
    model_path = 'models/person_classifier/thibault_model5/'
    model = CNN_thibault2()
    person_image_size = 50
    person_classifier = PersonClassifier(model_path, model, person_image_size)

    # set the thresholds
    prob_th = 0.9
    dist_th = 50
    target_counter_th = 3

    # initialize some variables
    target_counter = 0
    target_center = (0,0)
    past_target_center = (0,0)
    tracking_flag = False
    frame_counter = 0
    detection_counter = 0
    correct_counter = 0

    # prepare status information
    CNN_message = 'Searching with YOLOv2...'
    tracking_message = 'Tracking with KCF...'
    CNN_message_color = (255, 0, 255)
    tracking_message_color = (0, 255, 127)
    tracking_display_color = (0, 255, 0)
    # tracking_message_color = (255,0,0)
    # tracking_display_color = (255,0,0)
    trajectory_length = 40

    message = CNN_message
    message_color = CNN_message_color

    # setup a display
    cv2.namedWindow("video", cv2.WINDOW_NORMAL)

    # main loop
    while(True):

        # e1 = cv2.getTickCount()

        ret, frame = cap.read()
        target_ = 0

        if ret is not True:
            break

        if tracking_flag:
            # tracking
            message = tracking_message
            message_color = tracking_message_color
            track, bbox = tracker.update(frame)
            if track:
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                target_center = (int(bbox[0]+bbox[2]*0.5),int(bbox[1]+bbox[3]*0.5))

                # cv2.rectangle(frame, p1, p2, (49, 78, 234), 2, 1)
                cv2.rectangle(frame, p1, p2, tracking_display_color, 2, 1)

                cv2.circle(frame, target_center, 5, (0, 215, 253), -1)
                cv2.putText(frame, 'TARGET', (int(bbox[0]), int(bbox[1]+bbox[3])+25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, tracking_display_color, 2)
                target_ = 1
                detection_counter += 1
                # (49, 78, 234)

                trajectory_points.appendleft(target_center)

                for m in range(1, len(trajectory_points)):
                    if trajectory_points[m - 1] is None or trajectory_points[m] is None:
                        continue
                    cv2.line(frame, trajectory_points[m-1], trajectory_points[m],
                             tracking_display_color, thickness=2)

            else:
                tracking_flag = False
        else:
            # searching
            message = CNN_message
            message_color = CNN_message_color

            # get the YOLO's detection result
            nms_results = coco_predictor(frame)

            for result in nms_results:

                left, top = result["box"].int_left_top()
                right, bottom = result["box"].int_right_bottom()
                color = (255, 0, 255)

                # quit the loop if not "Person"
                if result["class_id"] != 0:
                    continue

                # get the module type CNN's recognition result
                person_class, prob = person_classifier(frame[top:bottom, left:right])

                # quit the loop if not "TARGET" with prob_th
                if person_class != 1 or prob < prob_th:
                    result_info = 'OTHERS(%2d%%)' % ((1.0-prob)*100)
                    draw_result(frame, result_info, left, top, right, bottom, color)
                    continue

                # print("target center point is renewed")
                target_counter += 1

                # calculate the target center point
                w = right - left
                h = bottom - top
                past_target_center = target_center
                target_center = (left+int(w*0.5),top+int(h*0.5))

                # calculate the Euclidean distance between current target center and the past one
                dist = np.linalg.norm(np.asarray(past_target_center)-np.asarray(target_center))

                # run tracking if there is a target in continuous frame and at close point by past one
                if (target_counter > target_counter_th and dist < dist_th):
                    tracker = setup_tracker()
                    bbox = (left,top,w,h)
                    ok = tracker.init(frame, bbox)
                    trajectory_points = deque(maxlen=trajectory_length)
                    tracking_flag = True
                    target_counter = 0
                else:
                    color = (0,255,0)
                    result_info = 'TARGET(%2d%%)' % (prob*100)
                    cv2.circle(frame, target_center, 5, (0, 215, 253), -1)
                    draw_result(frame, result_info, left, top, right, bottom, color)
                    target_ = 1
                    detection_counter += 1

        # display the total recognition result
        info(frame, message, message_color)

        if label_list[frame_counter] == target_:
            correct_counter += 1

        cv2.putText(frame, "correct label: "+str(label_list[frame_counter]), (10,80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        cv2.putText(frame, "output label:  "+str(target_), (10,120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        cv2.putText(frame, "frame: "+str(frame_counter), (10,160),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        cv2.imshow("video", frame)

        frame_counter += 1

        # record the video
        if not args.save_name == False:
            rec.write(frame)

        # key input to quit this program
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            end_flag = True

        # e2 = cv2.getTickCount()
        # fps = cv2.getTickFrequency() / (e2 - e1)
        # print("output fps: " + str(fps))

    acc = (correct_counter/label_N)*100
    print("\n"+"acc: "+str(acc)+"\n")

    acc_img = np.tile(np.uint8([245,245,245]), (height, width,1))

    # cv2.putText(acc_img, "correct label: "+str(correct_label_counter), (25,50),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
    # cv2.putText(acc_img, "detected label: "+str(detection_counter), (25,100),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    cv2.putText(acc_img, "accuracy: "+str(round(acc,2))+" %", (25,150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

    for i in range(90):
        cv2.imshow("video", acc_img)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if not args.save_name == False:
            rec.write(acc_img)

    # end processing
    cap.release()
    if not args.save_name == False:
        rec.release()
