import argparse
import numpy as np
import matplotlib.pyplot as plt

import cv2


def video(video_path, label_list):

    cap = cv2.VideoCapture(video_path)

    cnt = 0

    cv2.namedWindow('video_frame', cv2.WINDOW_NORMAL)

    while(True):
        ret, frame = cap.read()
        key = cv2.waitKey(25) & 0xFF

        if ret is not True:
            break

        cv2.putText(frame, "label: "+str(label_list[cnt]), (10,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,0,0), 2)
        cv2.putText(frame, "frame: "+str(cnt), (10,80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 2)
        cv2.imshow('video_frame', frame)

        cnt += 1

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='point tracking')
    parser.add_argument('--video_file', '-v', type=str, default=False,help='path to video')
    parser.add_argument('--label_file', '-l', type=str, default=False,help='path to label')
    args = parser.parse_args()

    video_path = args.video_file
    label_path = 'labels/' + args.label_file
    label_file = open(label_path,'r')
    label_list = []

    for label in label_file:
        label_list.append(int(label.split('\n')[0]))

    # print(label_list)
    print(len(label_list))

    video(video_path, label_list)
