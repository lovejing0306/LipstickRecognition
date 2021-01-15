# coding=utf-8

import dlib
import cv2
import numpy as np


class MouthDetection(object):
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('./model/shape_predictor_68_face_landmarks.dat')

    def get_mouth_keypoints(self, image):
        pos = np.zeros((3, 2), dtype=np.int8)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)

        dets = self.detector(image, 1)
        # 有些问题
        for index, face in enumerate(dets):
            shape = self.predictor(gray, face)
            points = shape.parts()
            pos[0][0] = points[56].x
            pos[0][1] = points[56].y
            pos[1][0] = points[57].x
            pos[1][1] = points[57].y
            pos[2][0] = points[58].x
            pos[2][1] = points[58].y
        return pos

    def crop(self, image, pos):
        images = list()
        x1 = pos[2][0]
        y1 = pos[2][1]
        x2 = pos[1][0]
        y2 = pos[1][1]
        d = abs(x2 - x1)
        images.append(image[(int)(y1 - d * 0.75):y2, x1:x2])

        x1 = pos[1][0]
        y1 = pos[1][1]
        x2 = pos[0][0]
        y2 = pos[0][1]
        d = abs(x1 - x2)
        images.append(image[y1 - d:y2, x1:x2])
        return images

    def get_mouth_images(self, image):
        pos = self.get_mouth_keypoints(image.copy())
        return self.crop(image, pos)