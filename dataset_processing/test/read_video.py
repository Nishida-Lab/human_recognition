import numpy as np
import matplotlib.pyplot as plt

import cv2


def video_to_images(video_path, image_path):

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(5)
    print fps

    cnt = 0

    cv2.namedWindow('video_frame', cv2.WINDOW_NORMAL)
    cv2.namedWindow('saved_frame', cv2.WINDOW_NORMAL)

    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret:

            cv2.imshow('video_frame', frame)
            cnt += 1

            key = cv2.waitKey(25) & 0xFF

            if key == ord('a'):
                print cnt

            if key == ord('s'):
                cv2.imshow('saved_frame', frame)
                cv2.imwrite(image_path+'im'+str(cnt)+'.jpg', frame)

            if key == ord('q'):
                break

    print cnt

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':

    video_path = '../../../dataset/video/cource_2017.mp4'
    image_path = '../../../dataset/images/from_video/'

    print video_path

    video_to_images(video_path, image_path)
