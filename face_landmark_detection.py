# coding=utf-8

import numpy as np
import cv2
import dlib
import requests
import json
import math


def polygons_to_mask(shape, regions_list):
    """
    polygons: labelme JSON中的边界点格式 [[x1,y1],[x2,y2],[x3,y3],...[xn,yn]]
    """
    mask = np.zeros(shape, dtype=np.uint8)
    for polygons in regions_list:
        cv2.fillPoly(mask, polygons, 1)  # 非int32 会报错
        # for polygon in polygons:
        #     polygon = np.asarray([polygon], np.int32)  # 这里必须是int32，其他类型使用fillPoly会报错
        #     cv2.fillPoly(mask, polygon, 1) # 非int32 会报错
        #     # cv2.fillConvexPoly(mask, polygon, 1)  # 非int32 会报错
    return mask


def draw_keypoints(image, landmarks_list):
    for landmarks in landmarks_list:
        for idx, point in enumerate(landmarks):
            # 68点的坐标
            pos = (point[0], point[1])
            # 利用cv2.circle给每个特征点画一个圈，共68个
            cv2.circle(img=image, center=pos, radius=2, color=(0, 255, 0), thickness=-1)
            # 利用cv2.putText输出1-68
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, str(idx + 1), pos, font, 0.3, (0, 0, 255), 1, cv2.LINE_AA)
    return image


def get_interpolation_point(pt1, pt2, dir, s):
    """
    pt1:第一个点的坐标
    pt2:第二个点的坐标
    dir:插值的方向
    s:插值的步长
    """
    x1 = pt1[0]
    y1 = -pt1[1]
    x2 = pt2[0]
    y2 = -pt2[1]
    if abs(x2 - x1) == 0:
        return (x2, (y1 + y2) / 2)
    k = (y2 - y1) / (x2 - x1)
    xc = (x1 + x2) / 2
    yc = (y1 + y2) / 2
    if k > 0:
        if dir > 0:
            x = xc - s
            y = k * s + yc
        else:
            x = xc + s
            y = -k * s + yc
    else:
        if dir > 0:
            x = xc + s
            y = -k * s + yc
        else:
            x = xc - s
            y = k * s + yc
    print((x1, y1))
    print((x2, y2))
    print((xc, yc))
    print(k)
    print((math.ceil(x), -math.ceil(y)))
    return (math.ceil(x), -math.ceil(y))


def draw_regions(image, face_regions_list):
    mask = polygons_to_mask(image.shape[0:2], face_regions_list)
    mask_ = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    image_segment = image * mask_
    return image_segment


class FaceLandmarkDlib(object):
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('./model/shape_predictor_68_face_landmarks.dat')

    def detect(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        landmarks_list = list()
        rects = self.detector(gray, 0)
        for i in range(len(rects)):
            landmarks = np.array([[p.x, p.y] for p in self.predictor(image, rects[i]).parts()])
            landmarks_list.append(landmarks)
        return landmarks_list

    def segment(self, landmarks_list):
        face_regions = list()
        for landmarks in landmarks_list:
            brow_l = landmarks[17:22, :]
            brow_r = landmarks[22:27, :]
            eye_l = landmarks[36:42, :]
            eye_r = landmarks[42:48, :]
            nose = np.concatenate((landmarks[27:28, :], landmarks[31:36, :]), axis=0)
            mouth = landmarks[48:60, :]
            face_regions.append([brow_l, brow_r, eye_l, eye_r, nose, mouth])
        return face_regions

    def get_segment_face(self, image_path):
        image = cv2.imread(image_path)
        # 特征点脸
        landmarks_list = self.detect(image)

        # 分割脸
        face_regions_list = self.segment(landmarks_list)
        image_segment = draw_regions(image, face_regions_list)

        return image_segment

    def get_landmark_face(self, image_path):
        """获取带有关键点的人脸"""
        image = cv2.imread(image_path)
        landmarks = self.detect(image)
        image = draw_keypoints(image, landmarks)
        return image


class FaceLandmarkFacePP(object):
    def __init__(self):
        self.url = 'https://api-cn.faceplusplus.com/facepp/v3/detect'
        self.config = {
            'api_key': 'nAVqQSpQPFItkaxfqKB2ZRmGfOUbBWq8',
            'api_secret': 'Nq4D-jiyDJ9rIddwn6_AZP3zw3PaGcMU',
            'return_landmark': 2,
            'return_attributes': 'none'
        }

    def get_landmark_info(self):
        names = list()
        with open('./doc/fpp_landmark.txt', 'r') as file:
            for line in file.readlines():
                line = line.strip('\n')
                items = line.split('\t')
                names.append(items[0])
        return names

    def detect(self, image_path):
        file = {'image_file': open(image_path, 'rb')}
        res = requests.post(self.url, files=file, data=self.config)
        data = json.loads(res.text)
        return data

    def segment(self, data, is_augment=False):
        face_regions = list()
        for face in data['faces']:
            landmark = face['landmark']
            brow_l = list()
            x1 = landmark['left_eyebrow_left_corner']['x']
            y1 = landmark['left_eyebrow_left_corner']['y']
            x2 = landmark['left_eyebrow_upper_left_quarter']['x']
            y2 = landmark['left_eyebrow_upper_left_quarter']['y']
            x3 = landmark['left_eyebrow_upper_middle']['x']
            y3 = landmark['left_eyebrow_upper_middle']['y']
            x4 = landmark['left_eyebrow_upper_right_quarter']['x']
            y4 = landmark['left_eyebrow_upper_right_quarter']['y']
            x5 = landmark['left_eyebrow_upper_right_corner']['x']
            y5 = landmark['left_eyebrow_upper_right_corner']['y']
            x6 = landmark['left_eyebrow_lower_right_corner']['x']
            y6 = landmark['left_eyebrow_lower_right_corner']['y']
            x7 = landmark['left_eyebrow_lower_right_quarter']['x']
            y7 = landmark['left_eyebrow_lower_right_quarter']['y']
            x8 = landmark['left_eyebrow_lower_middle']['x']
            y8 = landmark['left_eyebrow_lower_middle']['y']
            x9 = landmark['left_eyebrow_lower_left_quarter']['x']
            y9 = landmark['left_eyebrow_lower_left_quarter']['y']

            brow_l.append([x1, y1])
            if is_augment:
                x, y = get_interpolation_point((x1, y1), (x2, y2), 1, 1.5)
                brow_l.append([x, y])
            brow_l.append([x2, y2])
            if is_augment:
                x, y = get_interpolation_point((x2, y2), (x3, y3), 1, 1.5)
                brow_l.append([x, y])
            brow_l.append([x3, y3])
            brow_l.append([x4, y4])
            brow_l.append([x5, y5])
            brow_l.append([x6, y6])
            brow_l.append([x7, y7])
            brow_l.append([x8, y8])
            brow_l.append([x9, y9])

            brow_r = list()
            x1 = landmark['right_eyebrow_upper_left_corner']['x']
            y1 = landmark['right_eyebrow_upper_left_corner']['y']
            x2 = landmark['right_eyebrow_upper_left_quarter']['x']
            y2 = landmark['right_eyebrow_upper_left_quarter']['y']
            x3 = landmark['right_eyebrow_upper_middle']['x']
            y3 = landmark['right_eyebrow_upper_middle']['y']
            x4 = landmark['right_eyebrow_upper_right_quarter']['x']
            y4 = landmark['right_eyebrow_upper_right_quarter']['y']
            x5 = landmark['right_eyebrow_right_corner']['x']
            y5 = landmark['right_eyebrow_right_corner']['y']
            x6 = landmark['right_eyebrow_lower_right_quarter']['x']
            y6 = landmark['right_eyebrow_lower_right_quarter']['y']
            x7 = landmark['right_eyebrow_lower_middle']['x']
            y7 = landmark['right_eyebrow_lower_middle']['y']
            x8 = landmark['right_eyebrow_lower_left_quarter']['x']
            y8 = landmark['right_eyebrow_lower_left_quarter']['y']
            x9 = landmark['right_eyebrow_lower_left_corner']['x']
            y9 = landmark['right_eyebrow_lower_left_corner']['y']

            brow_r.append([x1, y1])
            # if is_augment:
            #     x, y = get_interpolation_point((x1, y1), (x2, y2), 1, 1.5)
            #     brow_r.append([x, y])
            brow_r.append([x2, y2])
            if is_augment:
                x, y = get_interpolation_point((x2, y2), (x3, y3), 1, 1.0)
                brow_r.append([x, y])
            brow_r.append([x3, y3])
            if is_augment:
                x, y = get_interpolation_point((x3, y3), (x4, y4), 1, 1.0)
                brow_r.append([x, y])
            brow_r.append([x4, y4])
            if is_augment:
                x, y = get_interpolation_point((x4, y4), (x5, y5), 1, 1.5)
                brow_r.append([x, y])
            brow_r.append([x5, y5])
            brow_r.append([x6, y6])
            brow_r.append([x7, y7])
            brow_r.append([x8, y8])
            brow_r.append([x9, y9])

            nose = list()
            nose.append([landmark['nose_bridge1']['x'], landmark['nose_bridge1']['y']])
            nose.append([landmark['nose_left_contour1']['x'], landmark['nose_left_contour1']['y']])
            nose.append([landmark['nose_left_contour2']['x'], landmark['nose_left_contour2']['y']])
            nose.append([landmark['nose_left_contour3']['x'], landmark['nose_left_contour3']['y']])
            nose.append([landmark['nose_left_contour4']['x'], landmark['nose_left_contour4']['y']])
            nose.append([landmark['nose_left_contour5']['x'], landmark['nose_left_contour5']['y']])
            nose.append([landmark['nose_middle_contour']['x'], landmark['nose_middle_contour']['y']])
            nose.append([landmark['nose_right_contour5']['x'], landmark['nose_right_contour5']['y']])
            nose.append([landmark['nose_right_contour4']['x'], landmark['nose_right_contour4']['y']])
            nose.append([landmark['nose_right_contour3']['x'], landmark['nose_right_contour3']['y']])
            nose.append([landmark['nose_right_contour2']['x'], landmark['nose_right_contour2']['y']])
            nose.append([landmark['nose_right_contour1']['x'], landmark['nose_right_contour1']['y']])

            eye_l = list()
            eye_l.append([landmark['left_eye_left_corner']['x'], landmark['left_eye_left_corner']['y']])
            eye_l.append([landmark['left_eye_upper_left_quarter']['x'], landmark['left_eye_upper_left_quarter']['y']])
            eye_l.append([landmark['left_eye_top']['x'], landmark['left_eye_top']['y']])
            eye_l.append([landmark['left_eye_upper_right_quarter']['x'], landmark['left_eye_upper_right_quarter']['y']])
            eye_l.append([landmark['left_eye_right_corner']['x'], landmark['left_eye_right_corner']['y']])
            eye_l.append([landmark['left_eye_lower_right_quarter']['x'], landmark['left_eye_lower_right_quarter']['y']])
            eye_l.append([landmark['left_eye_bottom']['x'], landmark['left_eye_bottom']['y']])
            eye_l.append([landmark['left_eye_lower_left_quarter']['x'], landmark['left_eye_lower_left_quarter']['y']])

            eye_r = list()
            eye_r.append([landmark['right_eye_left_corner']['x'], landmark['right_eye_left_corner']['y']])
            eye_r.append([landmark['right_eye_upper_left_quarter']['x'], landmark['right_eye_upper_left_quarter']['y']])
            eye_r.append([landmark['right_eye_top']['x'], landmark['right_eye_top']['y']])
            eye_r.append(
                [landmark['right_eye_upper_right_quarter']['x'], landmark['right_eye_upper_right_quarter']['y']])
            eye_r.append([landmark['right_eye_right_corner']['x'], landmark['right_eye_right_corner']['y']])
            eye_r.append(
                [landmark['right_eye_lower_right_quarter']['x'], landmark['right_eye_lower_right_quarter']['y']])
            eye_r.append([landmark['right_eye_bottom']['x'], landmark['right_eye_bottom']['y']])
            eye_r.append([landmark['right_eye_lower_left_quarter']['x'], landmark['right_eye_lower_left_quarter']['y']])

            mouth = list()
            mouth.append([landmark['mouth_left_corner']['x'], landmark['mouth_left_corner']['y']])
            mouth.append(
                [landmark['mouth_upper_lip_left_contour2']['x'], landmark['mouth_upper_lip_left_contour2']['y']])
            mouth.append(
                [landmark['mouth_upper_lip_left_contour1']['x'], landmark['mouth_upper_lip_left_contour1']['y']])
            mouth.append([landmark['mouth_upper_lip_top']['x'], landmark['mouth_upper_lip_top']['y']])
            mouth.append(
                [landmark['mouth_upper_lip_right_contour1']['x'], landmark['mouth_upper_lip_right_contour1']['y']])
            mouth.append(
                [landmark['mouth_upper_lip_right_contour2']['x'], landmark['mouth_upper_lip_right_contour2']['y']])
            mouth.append([landmark['mouth_right_corner']['x'], landmark['mouth_right_corner']['y']])
            mouth.append(
                [landmark['mouth_lower_lip_right_contour2']['x'], landmark['mouth_lower_lip_right_contour2']['y']])
            mouth.append(
                [landmark['mouth_lower_lip_right_contour3']['x'], landmark['mouth_lower_lip_right_contour3']['y']])
            mouth.append([landmark['mouth_lower_lip_bottom']['x'], landmark['mouth_lower_lip_bottom']['y']])
            mouth.append(
                [landmark['mouth_lower_lip_left_contour3']['x'], landmark['mouth_lower_lip_left_contour3']['y']])
            mouth.append(
                [landmark['mouth_lower_lip_left_contour2']['x'], landmark['mouth_lower_lip_left_contour2']['y']])

            face_regions.append([np.array(brow_l),
                                 np.array(brow_r),
                                 np.array(eye_l),
                                 np.array(eye_r),
                                 np.array(nose),
                                 np.array(mouth)])
        return face_regions

    def get_segment_face(self, image_path, is_augment=False):
        image = cv2.imread(image_path)
        # 特征点脸
        data = self.detect(image_path)
        # image_landmark = draw_keypoints(image.copy(), landmarks_list)

        # 分割脸
        face_regions_list = self.segment(data, is_augment)
        image_segment = draw_regions(image, face_regions_list)
        return image_segment

    def get_landmark_face(self, image_path):
        image = cv2.imread(image_path)
        data = self.detect(image_path)
        landmarks = list()
        for face in data['faces']:
            points = list()
            landmark = face['landmark']
            for name in self.get_landmark_info():
                point = landmark[name]
                points.append(np.array([point['x'], point['y']], dtype=np.int32))
            landmarks.append(points)
        image = draw_keypoints(image, landmarks)
        return image


def main(image_path):
    fld = FaceLandmarkDlib()
    flf = FaceLandmarkFacePP()

    # image = cv2.imread(image_path)
    #
    # # image_segment_dlib = fld.get_segment_face(image_path)
    # # image_landmark_fpp = flf.get_landmark_face(image_path)
    # image_segment_fpp = flf.get_segment_face(image_path)
    # image_segment_fpp2 = flf.get_segment_face(image_path, is_augment=True)
    #
    # concats = list()
    # concats.append(image)
    # # concats.append(image_segment_dlib)
    # # concats.append(image_landmark_fpp)
    # concats.append(image_segment_fpp)
    # concats.append(image_segment_fpp2)
    #
    # # 拼接
    # image_concat = cv2.hconcat(concats)
    # cv2.imwrite('./result/8.png', image_concat)

    image = flf.get_landmark_face(image_path)
    cv2.imwrite('./result/8_landmark.png', image)


if __name__ == '__main__':
    image_path = './datasets/8.png'
    main(image_path)
