import os
import json
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt


image_p_list = ['0110059_Fan_Zhao_20170821091746532']
for image_p in image_p_list:
    img_p = os.path.join('./image', image_p + '.jpg')
    json_p = os.path.join('./check_jsons', image_p + '_jpg_Label.json')
    mask_p = os.path.join('./check_masks', image_p + '_jpg_label_mask_255.jpg')
    assert os.path.exists(img_p) and os.path.join(json_p) and os.path.join(mask_p)

    with open(json_p, 'r', encoding='utf-8') as jread:
        json_data = json.load(jread)
    origin_image = Image.open(img_p, 'r')
    show_img = np.array(origin_image, dtype=np.uint8) / 255

    # get curve, landmark data
    temp_curve = json_data['Models']['PolygonModel2']  # list   [index]['Points']
    curve = {temp['Label']: [[int(i[0]), int(i[1])] for i in temp['Points']] for temp in temp_curve
             if len(temp['Points']) > 2}
    landmark = json_data['Models']['LandMarkListModel']['Points'][0]['LabelList']
    # 将landmark转换为int，同时去除第三个维度
    landmark = {i['Label']: [int(i['Position'][0]), int(i['Position'][1])] for i in landmark}
    # !!!!!!!!! np会改变Image的通道排列顺序，Image，为cwh，np为hwc，一般为chw（cv等等） cv:hwc
    mask_img = Image.open(mask_p)  # Image： c, w, h
    mask_array = np.array(mask_img)  # np 会将shape变为 h, w, c

    poly_curve = np.zeros_like(mask_array)
    for i in range(197, 210):  # 上颌骨区域
        poly_curve[mask_array == i] = 1
    for i in range(250, 256):  # 下颌骨区域
        poly_curve[mask_array == i] = 1
    for label, points in curve.items():
        for j in range(len(points) - 1):
            cv2.line(poly_curve, points[j], points[j + 1], color=1, thickness=6)

    for i in range(poly_curve.shape[0]):
        for j in range(poly_curve.shape[1]):
            if poly_curve[i][j] != 0:
                show_img[i][j][0] = show_img[i][j][0] * 0.7 + poly_curve[i][j] * 0.3
    for point in landmark.values():
        cv2.circle(show_img, point, 4, color=(0, 1, 0), thickness=-1)

    plt.imshow(show_img)
    plt.title(image_p)
    plt.show()


