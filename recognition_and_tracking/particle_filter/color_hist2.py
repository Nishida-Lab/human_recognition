import numpy as np
import cv2
from time import clock
import sys
import argparse


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = True, help = "Path to the image")
    args = vars(ap.parse_args())

    image = cv2.imread(args["image"])
    cv2.imshow("image", image)

    hsv_map = np.zeros((180, 256, 3), np.uint8)
    h, s = np.indices(hsv_map.shape[:2])
    hsv_map[:,:,0] = h
    hsv_map[:,:,1] = s
    hsv_map[:,:,2] = 255
    hsv_map = cv2.cvtColor(hsv_map, cv2.COLOR_HSV2BGR)

    hist_scale = 10

    def set_scale(val):
        global hist_scale
        hist_scale = val

    cv2.createTrackbar('scale', 'hist', hist_scale, 32, set_scale)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # dark = hsv[...,2] < 32
    # hsv[dark] = 0
    h = cv2.calcHist( [hsv], [0, 1], None, [180, 256], [0, 180, 0, 256] )

    h = np.clip(h*0.005*hist_scale, 0, 1)
    vis = hsv_map*h[:,:,np.newaxis] / 255.0
    cv2.imshow('hist', vis)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
