#!/bin/sh
python yolov2_with_person_recognition_and_tracking_acc.py -v ../../test_videos/test.avi -l test.txt -s KCF_acc.avi
python yolov2_with_person_recognition_and_tracking_acc.py -v ../../test_videos/test1.avi -l test1.txt -s KCF_acc1.avi
python yolov2_with_person_recognition_and_tracking_acc.py -v ../../test_videos/test2.avi -l test2.txt -s KCF_acc2.avi
python yolov2_with_person_recognition_and_tracking_acc.py -v ../../test_videos/test3.avi -l test3.txt -s KCF_acc3.avi
python yolov2_with_person_recognition_and_tracking_acc.py -v ../../test_videos/test4.mp4 -l test4.txt -s KCF_acc4.avi
