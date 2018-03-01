import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import copy

import cv2


def count_target_frame(video_path, label_path):

    cap = cv2.VideoCapture(video_path)

    frame_list = []
    while(True):

        ret, frame = cap.read()
        if ret is not True:
            break
        frame_list.append(frame)

    frame_N = len(frame_list)
    print("frame: "+str(frame_N))

    i = 0

    cv2.namedWindow('video_frame', cv2.WINDOW_NORMAL)
    cv2.imshow('video_frame', frame_list[i])

    label_list = [0]*frame_N
    label = 0

    while(True):

        key = cv2.waitKey(25) & 0xFF

        display_frame = copy.deepcopy(frame_list[i])

        if key == 83:
            i += 1

        if key == 81:
            i -= 1

        if key == ord('a'):
            label = 0

        if key == ord('s'):
            label = 1

        if key == ord('q') or i >= frame_N:
            break

        label_list[i] = label

        cv2.putText(display_frame, "label: "+str(label_list[i]), (10,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,0,0), 2)
        cv2.putText(display_frame, "frame: "+str(i), (10,80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 2)
        cv2.imshow('video_frame', display_frame)

    cap.release()
    cv2.destroyAllWindows()

    f = open(label_path,'w')
    for l in label_list:
        f.write(str(l) + '\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--video_file', '-v', type=str, default=False,help='path to video')
    parser.add_argument('--label_file', '-l', type=str, default=False,help='path to label')
    args = parser.parse_args()

    video_path = args.video_file
    label_path = 'labels/'

    if not os.path.isdir(label_path):
        os.makedirs(label_path)

    label_path = 'labels/' + args.label_file

    print(video_path)

    count_target_frame(video_path, label_path)
