# coding=utf-8

import json
import numpy as np
import os


class Lipstick(object):
    def __init__(self, lipstick_brand_path):
        self.lipstick_brand_path = lipstick_brand_path
        self.load_color()

    def convert_hexadecimal_to_rgb(self, s):
        assert len(s) == 6, 'the hexadecimal color is error.'
        values = list()
        for i in range(6):
            t = s[i]
            if t >= 'A' and t <= 'F':
                tt = ord(t) - ord('A') + 10
            else:
                tt = ord(t) - ord('0')
            values.append(tt)
        r = values[0] * 16 + values[1]
        g = values[2] * 16 + values[3]
        b = values[4] * 16 + values[5]
        return np.array([r, g, b])

    def load_color(self):
        assert os.path.exists(self.lipstick_brand_path), 'the lipstick brand is not exists.'
        colors = list()
        brands = list()
        with open(self.lipstick_brand_path, 'r') as f:
            data = json.load(f)
            for brand in data['brands']:
                brand_name = brand['name']
                for series in brand['series']:
                    series_name = series['name']
                    for lipstick in series['lipsticks']:
                        color = lipstick['color']
                        id = lipstick['id']
                        name = lipstick['name']
                        rgb = self.convert_hexadecimal_to_rgb(color[1:])
                        colors.append(rgb)
                        items = dict()
                        items['brand_name'] = brand_name
                        items['series_name'] = series_name
                        items['color'] = color
                        items['id'] = id
                        items['name'] = name
                        brands.append(items)
        self.colors = np.array(colors, dtype=np.int16)
        self.brands = brands

    def get_ids(self, color, top_k):
        color[1] = color[1] * 3 / 4
        diff = color - self.colors
        s = np.sum(np.abs(diff), axis=1)
        sort = np.argsort(s)
        if top_k > len(sort):
            top_k = len(sort)
        return sort[0:top_k]

    def get_brands(self, ids):
        brands = list()
        for id in ids:
            brands.append(self.brands[id])
        return brands

    def get_colors(self, ids):
        return self.colors[ids]