import cv2
import numpy as np
import copy
from collections import deque


class ParticleFilter:

    def __init__(self, image_size):

        self.SAMPLEMAX = 200
        self.height = image_size[0]
        self.width = image_size[1]

    def initialize(self):
        self.Y = np.random.random(self.SAMPLEMAX) * self.height
        self.X = np.random.random(self.SAMPLEMAX) * self.width

    def modeling(self):

        self.Y += np.random.random(self.SAMPLEMAX) * 100 - 50 # 2:1
        self.X += np.random.random(self.SAMPLEMAX) * 100 - 50

    def normalize(self, weight):
        return weight / np.sum(weight)

    def resampling(self, weight):
        index = np.arange(self.SAMPLEMAX)
        sample = []

        for i in range(self.SAMPLEMAX):
            idx = np.random.choice(index, p=weight)
            sample.append(idx)
        return sample

    def calcLikelihood(self, target_center):

        # sigma = 15.0
        # sigma = 5.0
        sigma = 200
        dist = []

        for i in range(self.SAMPLEMAX):
            y, x = self.Y[i], self.X[i]
            if y >= 0 and y < self.height and x >= 0 and x < self.width:
                d = (int(y-target_center[1])**2) + (int(x-target_center[0])**2)
                # print("i: " + str(i) + " x: "+str(x)+" y: "+str(y)+" ")
                dist.append(d)
            else:
                dist.append(-1)

        dist = np.array(dist)
        weights = (1.0 / np.sqrt(2.0 * np.pi * sigma)) * np.exp((-dist*dist)/(2.0*(sigma**2)))
        weights[dist == -1] = 0
        weights = self.normalize(weights)
        return weights

    def filtering(self, image):
        self.modeling()
        weights = self.calcLikelihood(image)
        index = self.resampling(weights)
        self.Y = self.Y[index]
        self.X = self.X[index]

        return np.sum(self.Y) / float(len(self.Y)), np.sum(self.X) / float(len(self.X))


class RunParticleFilter:

        def __init__(self, pf):

            self.pf = pf

            self.object_size = 400
            self.distance_th = 30 # 15

            trajectory_length = 20
            self.trajectory_points = deque(maxlen=trajectory_length)

        def clear(self):

            self.past_center = (0,0)
            self.center = (0,0)
            self.PF_start_flag = False
            self.pf.initialize()

        def RUN_PF(self, frame, target_center):

            cv2.circle(frame, target_center, 2, (0,0,255), -1)

            y, x = self.pf.filtering(target_center)

            frame_size = frame.shape
            p_range_x = np.max(self.pf.X)-np.min(self.pf.X)
            p_range_y = np.max(self.pf.Y)-np.min(self.pf.Y)

            for i in range(self.pf.SAMPLEMAX):
                # print("i: "+str(i)+" x: "+str(int(self.pf.X[i]))+" y: "+str(int(self.pf.Y[i])))
                cv2.circle(frame, (int(self.pf.X[i]), int(self.pf.Y[i])), 2, (203,192,255), -1)

            if p_range_x < self.object_size or p_range_y < self.object_size:

                self.past_center = self.center
                self.center = (int(x), int(y))

                if self.PF_start_flag is False:
                    self.past_center = target_center
                    self.PF_start_flag = True

                dist = np.linalg.norm(np.asarray(self.past_center)-np.asarray(self.center))

                print(dist)

                if self.PF_start_flag is True and dist > self.distance_th:
                    print("stop PF!: out of distance_th")
                    return frame,  False #PF_flag

                cv2.circle(frame, self.center, 5, (180,105,255), -1)
                self.trajectory_points.appendleft(self.center)

                for m in range(1, len(self.trajectory_points)):
                    if self.trajectory_points[m - 1] is None or self.trajectory_points[m] is None:
                        continue
                    cv2.line(frame, self.trajectory_points[m-1], self.trajectory_points[m],
                             (147,20,255), thickness=2)
            else:
                print("stop PF!: diverged")
                return frame, False #PF_flag

            return frame, True #PF_flag
