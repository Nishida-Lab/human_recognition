# -*- coding: utf-8 -*-

# Python libraries
import os
import numpy as np
from numpy.random import *
from PIL import Image
from PIL import ImageOps
import math
import random
import argparse

# OpenCV
import cv2


# overlay on part
def overlay_on_part(src_image, overlay_image, posX, posY):

    ol_height, ol_width = overlay_image.shape[:2]

    src_image_RGBA = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)
    overlay_image_RGBA = cv2.cvtColor(overlay_image, cv2.COLOR_BGRA2RGBA)

    src_image_PIL = Image.fromarray(src_image_RGBA)
    overlay_image_PIL = Image.fromarray(overlay_image_RGBA)

    src_image_PIL = src_image_PIL.convert('RGBA')
    overlay_image_PIL = overlay_image_PIL.convert('RGBA')

    tmp = Image.new('RGBA', src_image_PIL.size, (255, 255, 255, 0))
    tmp.paste(overlay_image_PIL, (posX, posY), overlay_image_PIL)
    result = Image.alpha_composite(src_image_PIL, tmp)

    return  cv2.cvtColor(np.asarray(result), cv2.COLOR_RGBA2BGRA)


# rotate
def rotate(img_src, angle):

    img_src = cv2.cvtColor(img_src, cv2.COLOR_BGRA2RGBA)
    img_PIL = Image.fromarray(img_src)
    rot = img_PIL.rotate(angle, expand=True)

    crop = rot.split()[-1].getbbox()
    rot = rot.crop(crop)

    return cv2.cvtColor(np.asarray(rot), cv2.COLOR_RGBA2BGRA)


# gamma gamma
def gamma_correction(img_src, gamma):
    lookUpTable = np.zeros((256, 1), dtype = 'uint8')

    for i in range(256):
        lookUpTable[i][0] = 255 * pow(float(i) / 255, 1.0 / gamma)

    img_gamma = cv2.LUT(img_src, lookUpTable)

    return img_gamma


# high contrast and low contrast table
def cont():
    min_table = 10
    max_table = 230
    diff_table = max_table - min_table

    LUT_HC = np.arange(256, dtype = 'uint8' )
    LUT_LC = np.arange(256, dtype = 'uint8' )

    for i in range(0, min_table):
        LUT_HC[i] = 0

    for i in range(min_table, max_table):
        LUT_HC[i] = 255 * (i - min_table) / diff_table

    for i in range(max_table, 255):
        LUT_HC[i] = 255

    for i in range(256):
        LUT_LC[i] = min_table + i * (diff_table) / 255

    return LUT_HC, LUT_LC


# mouse event detection
def mouse_event(event, x, y, flags, param):
    global background_img, human_img, overlapped_img, m_x, m_y

    if event == cv2.EVENT_LBUTTONUP:
        m_x = x
        m_y = y


# modify the size of rectangle
def check_rec_size(background_img, x, y, width, height):

    background_height, background_width = background_img.shape[:2]

    x1 = x - width/2
    y1 = y - height/2

    x2 = x + width/2
    y2 = y + height/2

    if x1 < 0: x1 = 0
    if y1 < 0: y1 = 0

    if x2 > background_width: x2 = background_width
    if y2 > background_height: y2 = background_height

    center_x = (x1+x2)/2
    center_y = (y1+y2)/2

    rec_width = x2-x1
    rec_height = y2 -y1

    center_x_dst = abs(center_x-background_width/2)

    if center_x > background_width/2:
        flipped_center_x = center_x - center_x_dst*2
    else:
        flipped_center_x = center_x + center_x_dst*2

    return center_x, flipped_center_x, center_y, rec_width, rec_height


# save rec list
def save_rec_list(rec_list,file_address_name):

    f = open(file_address_name,'w')
    for rec in rec_list:
        f.write(str(rec[0]) + ',' + str(rec[1]) + ',' + \
                str(rec[2]) + ',' + str(rec[3]) + '\n')


# save label list
def save_label_list(label_list,file_address_name):

    f = open(file_address_name,'w')
    for label in label_list:
        f.write(str(label) + '\n')


# purint the number of images
def how_many_images(orange_cnt, blue_cnt, background_cnt, img_cnt):

    print '\n' + \
          'the number of images \n' \
          '  orange image : ' + str(orange_cnt) +',' \
          '  blue image : ' + str(blue_cnt) +',' \
          '  background image : ' + str(background_cnt) +'\n' \
          'total image : ' + str(img_cnt) +'\n'


# overlap the human image to the background image
def creating_dataset(background_image_path, human_image_path, save_image_path):
    global background_img, human_img, overlapped_img, m_x, m_y

    img_cnt = 0
    human_cnt = 0
    orange_cnt = 0
    blue_cnt = 0
    background_cnt = 0
    label_list = []
    rec_list = []

    LUT_HC, LUT_LC = cont()

    while (True):

        key = cv2.waitKey(1)
        end_flag = False

        if human_cnt % 2 == 0:
            human_color = 'orange'
        else:
            human_color = 'blue'

        # read background image
        background_files = os.listdir(background_image_path)
        background_file_index = randint(len(background_files))
        background_img = cv2.imread(background_image_path+ \
                                    background_files[background_file_index])

        # read human image
        human_files = os.listdir(human_image_path+human_color+'/')
        human_file_index = randint(len(human_files))
        human_img = cv2.imread(human_image_path+human_color+'/'+ \
                               human_files[human_file_index], cv2.IMREAD_UNCHANGED)

        overlapped_img = background_img
        modified_human_img = human_img
        height, width = human_img.shape[:2]

        m_x = 100
        m_y = 100

        rate = 1.05

        cv2.namedWindow('overlapping', cv2.WINDOW_NORMAL)
        cv2.imshow('overlapping', background_img)
        cv2.setMouseCallback('overlapping', mouse_event)

        cv2.namedWindow('saved dataset', cv2.WINDOW_NORMAL)
        cv2.namedWindow('saved dataset (fliped)', cv2.WINDOW_NORMAL)

        while (True):

            key = cv2.waitKey(1)

            # expansion human image
            if key & 0xFF == ord("p"):
                height = int(height*rate)
                width = int(width*rate)
                modified_human_img = cv2.resize(modified_human_img, \
                                     (width,height),interpolation = cv2.INTER_LINEAR)

            # shrink human image
            if key & 0xFF == ord("n"):
                height = int(height*1./rate)
                width = int(width*1./rate)
                modified_human_img = cv2.resize(modified_human_img, \
                                     (width,height),interpolation = cv2.INTER_AREA)

            # make human image wide
            if key & 0xFF == ord("w"):
                width += 5
                modified_human_img = cv2.resize(modified_human_img, \
                                     (width,height),interpolation = cv2.INTER_LINEAR)

            # make human image tall
            if key & 0xFF == ord("h"):
                height += 5
                modified_human_img = cv2.resize(modified_human_img, \
                                     (width,height),interpolation = cv2.INTER_LINEAR)

            # flip human image
            if key & 0xFF == ord("f"):
                modified_human_img = cv2.flip(modified_human_img,1)

            # rotate human image left
            if key & 0xFF == ord("l"):
                modified_human_img = rotate(modified_human_img, 5)

            # rotate human image right
            if key & 0xFF == ord("r"):
                modified_human_img = rotate(modified_human_img, -5)

            # gamma_correction (high brightness)
            if key & 0xFF == ord("g"):
                modified_human_img = gamma_correction(modified_human_img, 1.05)

            # gamma_correction (high brightness)
            if key & 0xFF == ord("d"):
                modified_human_img = gamma_correction(modified_human_img, 0.95)

            # gamma_correction (high contrast)
            if key & 0xFF == ord("z"):
                modified_human_img = cv2.LUT(modified_human_img, LUT_HC)

            # gamma_correction (low contrast)
            if key & 0xFF == ord("x"):
                modified_human_img = cv2.LUT(modified_human_img, LUT_LC)

            # smoothing (average filter)
            if key & 0xFF == ord("a"):
                modified_human_img = cv2.blur(modified_human_img, (3,3))

            # reset human image
            if key & 0xFF == ord("o"):
                modified_human_img = human_img

            # save overlapped image
            if key & 0xFF == ord("s"):

                if human_color == 'orange':
                    orange_cnt += 1
                    label = 1
                    rec_color = (0,165,255)
                    print 'image ' +str(img_cnt)+ ' : ' + str(orange_cnt) + \
                    ' orange_human image + background image is saved !! (label:1)'

                elif human_color == 'blue':
                    blue_cnt += 1
                    label = 2
                    rec_color = (255,125,86)
                    print 'image ' +str(img_cnt)+ ' : ' + str(blue_cnt) + \
                    ' blue_human_image + background image is saved !!  (label:2)'

                label_list.append(label)
                cv2.imwrite(save_image_path+str(img_cnt)+'.jpg', overlapped_img)
                flipped_overlapped_img = cv2.flip(overlapped_img,1)

                center_x, flipped_center_x, center_y, rec_width, rec_height = \
                    check_rec_size(background_img, m_x, m_y, \
                    modified_human_img.shape[1], modified_human_img.shape[0])

                dataset_img = overlapped_img

                cv2.rectangle(dataset_img, \
                              (center_x - rec_width/2, \
                               center_y - rec_height/2),\
                              (center_x + rec_width/2, \
                               center_y + rec_height/2),\
                              rec_color, 10)

                rec_list.append([center_x, center_y, rec_width, rec_height])

                cv2.circle(dataset_img, (center_x, center_y), 10, (86, 255, 92), -1)
                cv2.imshow('saved dataset', dataset_img)

                img_cnt += 1
                human_cnt += 1

                # save fliped image
                cv2.imwrite(save_image_path+str(img_cnt)+'.jpg', flipped_overlapped_img)
                rec_list.append([flipped_center_x, center_y, rec_width, rec_height])
                label_list.append(label)
                img_cnt += 1

                flipped_dataset_img = flipped_overlapped_img

                cv2.rectangle(flipped_dataset_img, \
                              (flipped_center_x - rec_width/2, \
                               center_y - rec_height/2),\
                              (flipped_center_x + rec_width/2, \
                               center_y + rec_height/2),\
                              rec_color, 10)

                cv2.circle(flipped_dataset_img, (flipped_center_x, center_y), 10, (86, 255, 92), -1)
                cv2.imshow('saved dataset (fliped)', flipped_dataset_img)

            # save background image
            if key & 0xFF == ord("b"):

                background_cnt += 1
                rec_list.append([0, 0, 0, 0])
                label_list.append(0)
                cv2.imwrite(save_image_path+str(img_cnt)+'.jpg', background_img)

                print 'image ' +str(img_cnt)+ ' : ' + str(background_cnt) + \
                      ' background image is saved !! (label:0)'

                dataset_img = background_img

                cv2.imshow('saved dataset', dataset_img)
                img_cnt += 1

                # save fliped image
                rec_list.append([0, 0, 0, 0])
                label_list.append(0)
                flipped_background_img = cv2.flip(background_img,1)
                cv2.imwrite(save_image_path+str(img_cnt)+'.jpg', flipped_background_img)
                img_cnt += 1
                cv2.imshow('saved dataset (fliped)', flipped_background_img)


            # show image counter
            if key & 0xFF == ord("c"):
                how_many_images(orange_cnt, blue_cnt, background_cnt, img_cnt)

            # load next image
            if key & 0xFF == ord("q"):
                break

            # finish this program
            if key & 0xFF == 27:
                end_flag = True
                break

            height, width = modified_human_img.shape[:2]
            overlapped_img = overlay_on_part(background_img, modified_human_img, \
                                             m_x-width/2, m_y-height/2)
            cv2.imshow('overlapping', overlapped_img)

        if end_flag == True:
            break

    how_many_images(orange_cnt, blue_cnt, background_cnt, img_cnt)
    # print label_list
    # print len(rec_list)
    # print rec_list[0]
    # print rec_list[1]
    # print rec_list[2]
    save_label_list(label_list, save_image_path+'labels.txt')
    save_rec_list(rec_list, save_image_path+'rectangles.txt')
    cv2.destroyAllWindows()

    print 'label list and rec list are saved!'


# print how to use this script
def captions():
    print '\n' + \
          'Left click: change the position of the human\n' \
          'key P : expand the human\n' \
          'key N: shrink the human\n' \
          'key W: make the human wide\n' \
          'key H: make the human tall\n' \
          'key F: flip the human\n' \
          'key L: rotate the human counterclockwise\n' \
          'key R: rotate the human clockwise\n' \
          'key G: gamma correction (high brightness)\n' \
          'key D: gamma correction (low brightness)\n' \
          'key Z: gamma correction (high contrast)\n' \
          'key X: gamma correction (low contrast)\n' \
          'key A: smoothing (average filter)\n' \
          'key O: reset the human\n' \
          'key S: save the overlapped image\n' \
          'key Q: load the next images\n' \
          'key C: confirm current number of the images\n' \
          'key Esc: close everything\n \n' \


#main
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='image dataset for training cnn')
    parser.add_argument('--background_image_path', '-b', type=str, default="data",help='background image path')
    parser.add_argument('--human_image_path', '-h', type=str, default="data",help='human image path')
    parser.add_argument('--save_path', '-s', type=str, default="data",help='save image path')
    args = parser.parse_args()

    background_image_path = args.background_image_path + '/'
    human_image_path = args.human_image_path + '/'
    save_image_path = args.save_path + '/'

    if not os.path.isdir(save_image_path):
        os.makedirs(save_image_path)

    captions()
    creating_dataset(background_image_path, human_image_path, save_image_path)
