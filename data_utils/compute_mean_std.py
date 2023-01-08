import json
import os

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from transforms import GetROI


def roi(json_dir, json_data, img):
    # get curve, landmark data
    temp_curve = json_data['Models']['PolygonModel2']  # list   [index]['Points']
    curve = []
    # 去除标curve时多标的点
    for temp in temp_curve:
        if len(temp['Points']) > 2:
            curve.append(temp)
    landmark = json_data['Models']['LandMarkListModel']['Points'][0]['LabelList']
    # 将landmark转换为int，同时去除第三个维度
    landmark = {i['Label']: np.array(i['Position'], dtype=np.int32)[:2] for i in landmark}
    poly_points = json_data['Polys'][0]['Shapes']
    # get polygon mask
    mask_path = os.path.join('./check_masks', json_dir.split('\\')[-1].split('.')[0] + '_mask.jpg')
    # !!!!!!!!! np会改变Image的通道排列顺序，Image，为cwh，np为hwc，一般为chw（cv等等） cv:hwc
    mask_img = Image.open(mask_path)  # Image： c, w, h

    mask_array = np.array(mask_img)  # np 会将shape变为 h, w, c

    # 生成poly_curve 图
    poly_curve = np.zeros_like(mask_array)
    for i in range(4):
        if i == 0 or i == 1:  # 两个区域
            poly_curve[mask_array == i + 4] = i + 1
        elif i == 2 or i == 3:  # 两条直线
            points = curve[i - 3]['Points']
            label = curve[i - 3]['Label']
            points_array = np.array(points, dtype=np.int32)[:, :2]
            for j in range(len(points_array) - 1):
                cv2.line(poly_curve, points_array[j], points_array[j + 1], color=label - 3, thickness=6)
    # poly_curve = Image.fromarray(poly_curve)
    poly_curve = torch.as_tensor(poly_curve)
    # 得到标注的ROI区域图像
    img, mask, landmark, curve = GetROI(border_size=30)(img, {'mask': poly_curve, 'landmark': landmark, 'curve': {}})
    return img


def main():
    img_channels = 1
    cumulative_mean = np.zeros(img_channels)
    cumulative_std = np.zeros(img_channels)
    height = []
    width = []
    height_max, height_min = 0, 1000
    width_max, width_min = 0, 1000
    g_roi = True

    # 从train.txt 中读取信息
    txt_path = 'train.txt'
    json_root = './check_jsons'
    img_root = './image'
    with open(txt_path) as read:
        train_list = [line.strip() for line in read.readlines() if len(line.strip()) > 0]

    print(len(train_list))
    for json_path in tqdm(train_list):
        json_str = open(os.path.join('./check_jsons', json_path), 'r', encoding='utf-8')
        json_data = json.load(json_str)
        json_str.close()
        img_name = json_data['FileInfo']['Name']
        img_path = os.path.join(img_root, img_name)
        img = Image.open(img_path)
        img = img.convert('L')
        if g_roi:
            img = roi(json_path, json_data, img)
        w, h = img.size
        height.append(h)
        width.append(w)
        if w > width_max:
            width_max = w
        if h > height_max:
            height_max = h
        height_min = height_min if h > height_min else h
        width_min = width_min if w > width_min else w

        img = np.array(img) / 255.
        # gray image
        cumulative_mean += img.mean()
        cumulative_std += img.std()
        # rgb image
        # img = img.reshape(-1, 3)
        # cumulative_mean += img.mean(axis=0)
        # cumulative_std += img.std(axis=0)

    # 生成所有的图片名
    # img_list = []
    # for name_split in [name.split('_')[:-2] for name in train_list]:
    #     str = './image/'
    #     for temp in name_split[:-1]:
    #         str += temp + '_'
    #     str += name_split[-1]
    #     if os.path.exists(str + '.jpg'):
    #         img_list.append(str + '.jpg')
    #     elif os.path.exists(str + '.JPG'):
    #         img_list.append(str + '.JPG')
    #     else:
    #         raise '{}.jpg/JPG does not exists'.format(str)

    mean = cumulative_mean / len(train_list)
    std = cumulative_std / len(train_list)
    print(f"mean: {mean}")
    print(f"std: {std}")
    print(f'average height : {np.mean(height)},    height std : {np.std(height)}')
    print(f'average width : {np.mean(width)},    width std : {np.std(width)}')
    print(f'max height : {height_max}   min height : {height_min}')
    print(f'max width : {width_max}   min width : {width_min}')
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
    main()
