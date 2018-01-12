import numpy as np
import cv2

def camshift(cap, rec, initial_left, initial_top, initial_width, initial_height):

    ret ,frame = cap.read()
    track_window = (initial_left,initial_top,initial_width,initial_height)

    # set up the ROI for tracking
    roi = frame[initial_top:initial_top+initial_height, initial_left:initial_left+initial_width]
    cv2.imshow("ROI", roi)
    hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # 60, 32
    #255
    mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
    roi_hist = cv2.calcHist([hsv_roi],[0],mask,[20],[0,180])
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

    # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

    # thresholds
    search_range_th = 15
    rec_size_th = 140
    dist_th = 30
    scale_th = 1.8

    target_center = (0,0)
    past_target_center = (0,0)

    color = (49, 78, 234)
    start_flag = False
    end_flag = False

    while(True):

        ret ,frame = cap.read()

        if ret == True:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

            # apply camshift to get the new location
            ret, track_window = cv2.CamShift(dst, track_window, term_crit)

            x,y,w,h = track_window

            target_center = (x+int(w/2),y+int(h/2))
            height, width = frame.shape[:2]

            # threshold process

            # size threshold
            if w > rec_size_th or h > rec_size_th:
                # print("over size")
                break

            # frame range threshold
            elif target_center[0] < search_range_th or target_center[1] < search_range_th or \
            target_center[0] > width-search_range_th or \
            target_center[1] > height-search_range_th:
                # print("over range")
                break

            # cener distance and rectangle scale threshold
            elif start_flag is True:
                dist = np.linalg.norm(np.asarray(past_target_center)-np.asarray(target_center))
                w_scale = w/past_w
                h_scale = h/past_h

                # print(w_scale)
                # print(h_scale)
                # print()

                if dist > dist_th:
                    # print("over dist")
                    break
                elif w_scale > scale_th and h_scale > scale_th:
                    # print("over scale")
                    break

            # Draw it on image
            tracking_result = cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
            cv2.circle(tracking_result, target_center, 2, (0, 215, 253), 2)
            cv2.putText(tracking_result, 'Tracking with Camshift ...', (10,18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(tracking_result, 'TARGET', (x,y+h+25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            # pts = cv2.boxPoints(ret)
            # pts = np.int0(pts)
            # img2 = cv2.polylines(frame,[pts],True, (0,255,0),2)
            cv2.imshow('video',tracking_result)

            k = cv2.waitKey(60) & 0xff
            if k == 27:
                end_flag = True

            past_target_center = target_center
            past_w = w
            past_h = h
            start_flag = True

            if not rec == False:
                rec.write(tracking_result)
        else:
            break

    return end_flag
