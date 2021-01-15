# coding=utf-8

from lipstick import Lipstick
from color_detection import ColorDetection
from mouth_detection import MouthDetection
import cv2
import numpy as np


class ColorRecognition(object):
    def __init__(self):
        self.li = Lipstick('./datasets/lipstick.json')
        self.cd = ColorDetection()
        self.md = MouthDetection()

    def recognize(self, image, top_k):
        images = self.md.get_mouth_images(image)
        color = self.cd.color_mean(images)
        # color = np.array([156, 59, 103])
        ids = self.li.get_ids(color, top_k)
        brands = self.li.get_brands(ids)
        colors = self.li.get_colors(ids)
        return brands, colors


if __name__ == '__main__':
    image_path = './datasets/women.jpg'
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cr =ColorRecognition()
    brands, colors = cr.recognize(image, 3)
    print(brands, colors)