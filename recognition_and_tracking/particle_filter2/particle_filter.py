import cv2
# import cv2.cv as cv
import numpy as np
import colorsys
from PIL import Image
import copy
from collections import deque


class ParticleFilter:

    def __init__(self, image_size):

        self.SAMPLEMAX = 300
        self.height = image_size[0]
        self.width = image_size[1]

    def initialize(self):
        self.Y = np.random.random(self.SAMPLEMAX) * self.height
        self.X = np.random.random(self.SAMPLEMAX) * self.width

    # Need adjustment for tracking object velocity
    def modeling(self):

        self.Y += np.random.random(self.SAMPLEMAX) * 40 - 20 # 2:1
        self.X += np.random.random(self.SAMPLEMAX) * 40 - 20

    def normalize(self, weight):
        return weight / np.sum(weight)

    def resampling(self, weight):
        index = np.arange(self.SAMPLEMAX)
        sample = []

        # choice by weight
        for i in range(self.SAMPLEMAX):
            idx = np.random.choice(index, p=weight)
            sample.append(idx)
        return sample

    def calcLikelihood(self, image):

        # white space tracking
        mean, std = 250.0, 15.0

        intensity = []

        for i in range(self.SAMPLEMAX):
            y, x = self.Y[i], self.X[i]
            if y >= 0 and y < self.height and x >= 0 and x < self.width:
                intensity.append(image[int(y),int(x)])
            else:
                intensity.append(-1)

        # normal distribution
        weights = 1.0 / np.sqrt(2 * np.pi * std) * np.exp(-(np.array(intensity) - mean)**2 /(2 * std**2))
        weights[intensity == -1] = 0
        weights = self.normalize(weights)
        return weights

    def filtering(self, image):
        self.modeling()
        weights = self.calcLikelihood(image)
        index = self.resampling(weights)
        self.Y = self.Y[index]
        self.X = self.X[index]

        # return COG
        return np.sum(self.Y) / float(len(self.Y)), np.sum(self.X) / float(len(self.X))


def RUN_PF(cap, rec, pf, _LOWER_COLOR, _UPPER_COLOR, dominant_bgr, high_bgr, crop_center):

    # PF
    object_size = 100
    distance_th = 15

    trajectory_length = 20
    trajectory_points = deque(maxlen=trajectory_length)

    PF_start = False
    center = (0,0)
    past_center = (0,0)

    while True:

        ret, frame = cap.read()
        result_frame = copy.deepcopy(frame)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Threshold the HSV image to get only a color
        mask = cv2.inRange(hsv, _LOWER_COLOR, _UPPER_COLOR)

        # Start Tracking
        y, x = pf.filtering(mask)

        frame_size = frame.shape
        p_range_x = np.max(pf.X)-np.min(pf.X)
        p_range_y = np.max(pf.Y)-np.min(pf.Y)

        for i in range(pf.SAMPLEMAX):
            cv2.circle(result_frame, (int(pf.X[i]), int(pf.Y[i])), 2, dominant_bgr, -1)

        # if p_range_x < object_size and p_range_y < object_size:
        if p_range_x < object_size or p_range_y < object_size:

            past_center = center
            center = (int(x), int(y))

            if PF_start is False:
                past_center = crop_center
                PF_start = True

            dist = np.linalg.norm(np.asarray(past_center)-np.asarray(center))

            print(dist)

            if PF_start is True and dist > distance_th:
                print("stop PF!: out of distance_th")
                return

            cv2.circle(result_frame, center, 5, (0, 215, 253), -1)
            trajectory_points.appendleft(center)

            for m in range(1, len(trajectory_points)):
                if trajectory_points[m - 1] is None or trajectory_points[m] is None:
                    continue
                cv2.line(result_frame, trajectory_points[m-1], trajectory_points[m], high_bgr, thickness=2)
        else:
            print("stop PF!: diverged")
            return

        cv2.putText(result_frame, 'Tracking with Particle Filter ...', (10,18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (204, 153, 51), 2)

        cv2.imshow("video", result_frame)

        if not rec == False:
            rec.write(result_frame)

        if cv2.waitKey(20) & 0xFF == 27:
            break


def get_dominant_color(image):
    """
    Find a PIL image's dominant color, returning an (r, g, b) tuple.
    """
    image = image.convert('RGBA')
    # Shrink the image, so we don't spend too long analysing color
    # frequencies. We're not interpolating so should be quick.
    image.thumbnail((200, 200))
    max_score = 0.0
    dominant_color = None

    for count, (r, g, b, a) in image.getcolors(image.size[0] * image.size[1]):
        # Skip 100% transparent pixels
        if a == 0:
            continue
        # Get color saturation, 0-1
        saturation = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)[1]
        # Calculate luminance - integer YUV conversion from
        # http://en.wikipedia.org/wiki/YUV
        y = min(abs(r * 2104 + g * 4130 + b * 802 + 4096 + 131072) >> 13, 235)
        # Rescale luminance from 16-235 to 0-1
        y = (y - 16.0) / (235 - 16)
        # Ignore the brightest colors
        if y > 0.9:
            continue
        # Calculate the score, preferring highly saturated colors.
        # Add 0.1 to the saturation so we don't completely ignore grayscale
        # colors by multiplying the count by zero, but still give them a low
        # weight.
        score = (saturation + 0.1) * count
        if score > max_score:
            max_score = score
            dominant_color = [b, g, r]

    return dominant_color


def bgr_to_hsv(bgr_color):
    hsv = cv2.cvtColor(np.array([[[bgr_color[0], bgr_color[1], bgr_color[2]]]],
                                dtype=np.uint8), cv2.COLOR_BGR2HSV)[0][0]
    return (int(hsv[0]), int(hsv[1]), int(hsv[2]))



def hsv_to_bgr(hsv_color):
    bgr = cv2.cvtColor(np.array([[[hsv_color[0], hsv_color[1], hsv_color[2]]]],
                                dtype=np.uint8),cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[0]), int(bgr[1]),int(bgr[2]))
