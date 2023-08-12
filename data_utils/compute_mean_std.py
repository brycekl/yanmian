import json
import os

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def roi(json_data, img, border_size=0, scale_factor=1):
    # get curve, landmark data
    temp_curve = json_data['Models']['PolygonModel2']  # list   [index]['Points']
    curve = []
    # 去除标curve时多标的点
    for temp in temp_curve:
        if len(temp['Points']) > 2:
            curve.append(temp)

    # 将landmark转换为int，同时去除第三个维度
    landmarks = np.zeros((6, 2))
    for i in json_data['Models']['LandMarkListModel']['Points'][0]['LabelList']:
        landmarks[i['Label']-8] = i['Position'][:2]

    # 获取曲线的点
    curves = {}
    for i in json_data['Models']['PolygonModel2']:
        curves[i['Label']] = np.array(i['Points'])[:, :2]

    # 分割轮廓点
    polys = {}
    for i in json_data['Polys'][0]['Shapes']:
        polys[i['labelType']] = np.array([j['Pos'][:2] for j in i['Points']])

    all_keypoints = np.vstack((landmarks, curves[6], curves[7], polys[4], polys[5]))
    min_x, min_y = all_keypoints.min(axis=0)
    max_x, max_y = all_keypoints.max(axis=0)

    box = change_box([min_x, min_y, max_x, max_y], *img.shape, scale_factor, border_size)
    min_x, min_y, max_x, max_y = box

    roi_img = img[min_y: max_y, min_x: max_x]
    return roi_img, box


def change_box(box, img_h, img_w,  scale, pad_size):
    min_x, min_y, max_x, max_y = box
    box_w, box_h = max_x-min_x, max_y-min_y
    scale_w, scale_h = box_w * (scale - 1) + 2 * pad_size, box_h * (scale - 1) + 2 * pad_size
    min_x, min_y = max(int(min_x - scale_w / 2 + 0.5), 0), max(int(min_y - scale_h / 2 + 0.5), 0)
    max_x, max_y = min(int(min_x + scale_w + 0.5 + box_w), img_w), min(int(min_y + scale_h + 0.5 + box_h), img_h)
    return [min_x, min_y, max_x, max_y]


def his(img):
    # image_root = '../datas/MMPose/1.png'
    # img = cv2.imread(image_root, 0)
    # 直方图均衡
    # equ = cv2.equalizeHist(img)
    # 限制对比度自适应直方图均衡化(CLAHE)  --> 效果更好
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(img)
    return cl1


def main():
    means, std, height, width = [], [], [], []
    g_roi = True
    do_his = False
    after_roi_his = False

    # 从train.txt 中读取信息
    txt_path = 'all.txt'
    json_root = '../datas/jsons'
    img_root = '../datas/images'
    with open(txt_path) as read:
        train_list = [line.strip() for line in read.readlines() if len(line.strip()) > 0]

    print(len(train_list))
    for json_path in tqdm(train_list):
        json_str = open(os.path.join(json_root, json_path), 'r', encoding='utf-8')
        json_data = json.load(json_str)
        json_str.close()
        img_name = json_data['FileInfo']['Name']
        img_path = os.path.join(img_root, img_name)
        img = Image.open(img_path)
        img = np.array(img.convert('L'))
        if do_his:
            img = his(img)
        if g_roi:
            img, roi_box = roi(json_data, img, border_size=30, scale_factor=1)
        if after_roi_his:
            img = his(img)
        h, w = img.shape
        height.append(h)
        width.append(w)

        img = np.array(img) / 255.
        # gray image
        means.append(img.mean())
        std.append(img.std())

    means, std = np.array(means), np.array(std)
    height, width = np.array(height), np.array(width)
    print(f"mean: {means.mean()}")
    print(f"std: {std.mean()}")
    print(f'average height : {height.mean()},    height std : {height.std()}')
    print(f'average width : {width.mean()},    width std : {width.std()}')
    print(f'max height : {height.max()}   min height : {height.min()}')
    print(f'max width : {width.max()}   min width : {width.min()}')

    # 聚类分析
    w_h = np.hstack((width.reshape(-1, 1), height.reshape(-1, 1)))
    y_pred = KMeans(n_clusters=2, random_state=1).fit_predict(w_h)
    plt.scatter(w_h[:, 0], w_h[:, 1], c=y_pred)
    plt.show()

    m_s = np.hstack((means.reshape(-1, 1), std.reshape(-1, 1)))
    y_pred = KMeans(n_clusters=2, random_state=1).fit_predict(m_s)
    plt.scatter(m_s[:, 0], m_s[:, 1], c=y_pred)
    plt.show()

    # mean: [0.22270182 0.22453914 0.22637838]  [0.22922716 0.22991307 0.23040238]    [0.22790766 0.22808619 0.22835823]
    # std: [0.21268971 0.21371627 0.21473691]   [0.21878482 0.21936761 0.21957956]    [0.23118967 0.23132948 0.23143777]
    # average height: 762.9803921568628     739.983651226158     353.67639429312584
    # average width: 1088.7843137254902    1040.2125340599455    393.01815823605705
    # max height: 866                                           581
    # max width: 1260                                           682(ROI)

    # check 3                                   data 5
    # mean : [0.2347, 0.2350, 0.2353]      [0.2356, 0.2359, 0.2362]
    # std : [0.2209, 0.2211, 0.2211]       [0.2210, 0.2212, 0.2212]


if __name__ == '__main__':
    # 注意分析不同的图像，有很大的区别（多中心）
    # 如图片大小（正中与半身），对比度，光照，灰度等（直方图均衡）
    main()
