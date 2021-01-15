# coding=utf-8

from PIL import Image
import colorsys
import numpy as np


class ColorDetection(object):
    def __init__(self):
        pass

    def get_dominant_color(self, image):
        image = Image.fromarray(np.uint8(image))
        max_score = 0.0001
        dominant_color = None
        for count, (r, g, b) in image.getcolors(image.size[0] * image.size[1]):
            # 转为HSV标准
            saturation = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)[1]
            y = min(abs(r * 2104 + g * 4130 + b * 802 + 4096 + 131072) >> 13, 235)
            y = (y - 16.0) / (235 - 16)

            # 忽略高亮色
            if y > 0.9:
                continue
            score = (saturation + 0.1) * count
            if score > max_score:
                max_score = score
                dominant_color = np.array([r, g, b], dtype=np.int16)
        return dominant_color

    def color_mean(self, images):
        colors = list()
        for image in images:
            color = self.get_dominant_color(image)
            if color is None:
                pass
            else:
                colors.append(color)
        colors = np.array(colors)
        return np.mean(colors, axis=0)