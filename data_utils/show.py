import os
import json
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from eva_utils.eval_metric import *


def show_image(image_p_list):
    # image_p_list = ['0110059_Fan_Zhao_20170821091746532']
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


def show_one_metric(img, line1, line2, keypoints, metric, value, value_point, line3=None, color=(0, 255, 0),
                    save_path=None, rewrite=False):
    copy_img = img.copy()
    img = np.asarray(img)
    cv2.line(img, line1[0], line1[1], color, thickness=3)
    cv2.line(img, line2[0], line2[1], color, thickness=3)
    if metric == 'IFA':
        cv2.line(img, line3[0], line3[1], color, thickness=3)

    if save_path:
        cv2.putText(img, value, value_point, cv2.FONT_HERSHEY_COMPLEX, 1.0, color=color, thickness=4)
        if metric in ['IFA', 'FMA']:
            cv2.putText(img, 'o', [value_point[0] + 70, value_point[1]-15], cv2.FONT_HERSHEY_COMPLEX, 0.5, color=color, thickness=2)
        if metric in ['MNM']:
            cv2.putText(img, 'o', [value_point[0] + 50, value_point[1] - 15], cv2.FONT_HERSHEY_COMPLEX, 0.5,
                        color=color, thickness=2)

    for keypoint in keypoints:
        cv2.circle(img, keypoint, 4, (255, 0, 0), -1)

    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.imshow(img)
        plt.show()
        save_img = Image.fromarray(img)
        save_img.save(os.path.join(save_path, f'./{metric}.png'))
    return img if rewrite else copy_img


def show_metrics(img):
    upper_lip = [700, 270]
    under_lip = [705, 303]
    upper_midpoint = [649, 289]
    under_midpoint = [695, 342]
    chin = [728, 358]
    nasion = [567, 227]
    img = show_one_metric(img, [nasion, [558, 182]], [[438, 245], [813, 193]], [nasion, upper_lip, under_lip, chin],
                    'IFA', str(98.5), [600, 259], [chin, [682, 210]], color=(173, 255, 47), save_path='./')
    img = show_one_metric(img, [upper_midpoint, nasion], [under_midpoint, nasion], [upper_midpoint, under_midpoint, nasion],
                          'MNM', str(4.9), [606, 249], color=(255, 255, 0), save_path='./')
    img = show_one_metric(img, [[444, 389], [773, 196]], [chin, [691, 243]], [chin, upper_lip],
                    'FMA', str(77.3), [623, 310], (255, 250, 205), save_path='./')
    img = show_one_metric(img, [under_midpoint, [414, 89]], [[543, 141], [508, 173]], [under_midpoint, nasion],
                    'PL', str(round(0.005555555690 * 10 * 48.7, 2)) + 'mm', [527, 181], color=(0, 255, 255), save_path='./')
    img = show_one_metric(img, [under_midpoint, [458, 68]], [[543, 141], [530, 152]], [under_midpoint, upper_midpoint],
                    'FS', str(round(0.005555555690 * 10 * 17.8, 2)) + 'mm', [545, 159], color=(255, 0, 255), save_path='./')

    # 绘制所有图
    img = show_one_metric(img, [nasion, [558, 182]], [[438, 245], [813, 193]], [nasion, upper_lip, under_lip, chin],
                    'IFA', str(98.5), [600, 259], [chin, [682, 210]], color=(173, 255, 47), rewrite=True)
    img = show_one_metric(img, [upper_midpoint, nasion], [under_midpoint, nasion],
                          [upper_midpoint, under_midpoint, nasion], 'MNM', str(4.9), [606, 249], color=(255, 255, 0), rewrite=True)
    img = show_one_metric(img, [[444, 389], [773, 196]], [chin, [691, 243]], [chin, upper_lip],
                    'FMA', str(77.3), [623, 310], (255, 250, 205), rewrite=True)
    img = show_one_metric(img, [under_midpoint, [414, 89]], [[543, 141], [508, 173]], [under_midpoint, nasion],
                    'PL', str(round(0.005555555690 * 10 * 48.7, 2)) + 'mm', [527, 181], color=(0, 255, 255), rewrite=True)
    img = show_one_metric(img, [under_midpoint, [458, 68]], [[543, 141], [530, 152]], [under_midpoint, upper_midpoint],
                    'FS', str(round(0.005555555690 * 10 * 17.8, 2)) + 'mm', [545, 159], color=(255, 0, 255), rewrite=True)

    img = np.asarray(img)
    x = -40
    img[img.shape[0] - 255 + x:img.shape[0] - 70 + x, :250, :] = 0
    cv2.putText(img, 'IFA: ' + str(round(98.5, 2)), [20, img.shape[0] - 220 + x], cv2.FONT_HERSHEY_COMPLEX, 1.0,
                (173, 255, 47), 2)
    ind = np.where(img[img.shape[0] - 240 + x:img.shape[0] - 220 + x, :250, 0] != 0)[1].max()
    cv2.putText(img, 'o', [ind + 5, img.shape[0] - 232 + x], cv2.FONT_HERSHEY_COMPLEX, 0.5,
                (173, 255, 47), 2)
    cv2.putText(img, 'MNM: ' + str(round(4.9, 2)), [20, img.shape[0] - 185 + x], cv2.FONT_HERSHEY_COMPLEX, 1.0,
                (255, 255, 0), 2)
    ind = np.where(img[img.shape[0] - 205 + x:img.shape[0] - 185 + x, :250, 0] != 0)[1].max()
    cv2.putText(img, 'o', [ind + 5, img.shape[0] - 197 + x], cv2.FONT_HERSHEY_COMPLEX, 0.5,
                (255, 255, 0), 2)
    cv2.putText(img, 'FMA: ' + str(round(77.3, 2)), [20, img.shape[0] - 150 + x], cv2.FONT_HERSHEY_COMPLEX, 1.0,
                (255, 250, 205), 2)
    ind = np.where(img[img.shape[0] - 170 + x:img.shape[0] - 150 + x, :250, 0] != 0)[1].max()
    cv2.putText(img, 'o', [ind + 5, img.shape[0] - 162 + x], cv2.FONT_HERSHEY_COMPLEX, 0.5,
                (255, 250, 205), 2)
    cv2.putText(img, 'FS: ' + str(round(0.99, 2)) + 'mm', [20, img.shape[0] - 115 + x], cv2.FONT_HERSHEY_COMPLEX,
                1.0, (255, 0, 255), 2)
    cv2.putText(img, 'PL: ' + str(round(2.71, 2)) + 'mm', [20, img.shape[0] - 80 + x], cv2.FONT_HERSHEY_COMPLEX,
                1.0, (0, 255, 255), 2)
    plt.imshow(img)
    plt.show()
    img = Image.fromarray(img)
    img.save('./total.png')


def show_all_metrics(img):
    upper_lip = [700, 270]
    under_lip = [705, 303]
    upper_midpoint = [649, 289]
    under_midpoint = [695, 342]
    chin = [728, 358]
    nasion = [567, 227]
    show_one_metric(img, [nasion, [558, 182]], [[438, 245], [813, 193]], [nasion, upper_lip, under_lip, chin],
                    'IFA', str(98.5), [600, 259], [chin, [682, 210]], color=(173, 255, 47))
    show_one_metric(img, [upper_midpoint, nasion], [under_midpoint, nasion], [upper_midpoint, under_midpoint, nasion],
                    'MNM', str(4.9), [606, 249], color=(255, 255, 0))
    show_one_metric(img, [[444, 389], [773, 196]], [chin, [691, 243]], [chin, upper_lip],
                    'FMA', str(77.3), [623, 310], (255, 250, 205))
    show_one_metric(img, [under_midpoint, [414, 89]], [[543, 141], [508, 173]], [under_midpoint, nasion],
                    'PL', str(round(0.005555555690 * 10 * 48.7, 2)) + 'mm', [527, 181], color=(0, 255, 255))
    show_one_metric(img, [under_midpoint, [458, 68]], [[543, 141], [530, 152]], [under_midpoint, upper_midpoint],
                    'FS', str(round(0.005555555690 * 10 * 17.8, 2)) + 'mm', [545, 159], color=(255, 0, 255))


def get_preg_data():
    """
    获取孕周，年龄信息
    """
    d = pd.read_excel('./data_information/image_info.xlsx', sheet_name=None)
    data = {i: j.to_dict('list') for i, j in d.items()}

    date = pd.read_excel('./data_information/孕妇年龄和孕周.xlsx')
    date = date.to_dict('list')
    date['label'] = date['Unnamed: 0']
    for label in data:
        for i, lab in enumerate(data[label]['name']):
            lab_ = None
            if lab[:7] in date['label']:
                lab_ = lab[:7]
            elif lab[1:7] in date['label'] and lab[0] == '0':
                lab_ = lab[1:7]
            if lab_:
                ind = date['label'].index(lab_)
                if isinstance(date['年龄'][ind], str):
                    year = date['年龄'][ind].split('岁')[0]
                    data[label]['year'][i] = year
                if isinstance(date['超声孕周'][ind], str):
                    week = date['超声孕周'][ind].split('/')[0]
                    data[label]['week'][i] = week.split('W')[0].split('w')[0]
                    data[label]['day'][i] = week.split('W')[-1].split('w')[-1].split('D')[0].split('d')[0]

    # 统计没有孕周的数据
    dd = {'year': 0, 'week': 0, 'day': 0, 'name': []}
    for i in range(len(data['normal']['year'])):
        for j in ['year', 'week', 'day']:
            if not isinstance(data['normal'][j][i], str):
                dd[j] += 1
                dd['name'].append(data['normal']['name'][i])


if __name__ == "__main__":
    img = Image.open('../data/images/35300920211104_yuanqing_gao_20211104102136230.jpg')
    show_metrics(img)

