import numpy as np
from PIL import Image
import cv2
import torch


def get_anno(json_data, landmark_num=6):
    # 获取json里标注的关键点和曲线信息
    temp_curve = json_data['Models']['PolygonModel2']  # list   [index]['Points']
    curve = []
    # 去除标curve时多标的点
    for temp in temp_curve:
        if len(temp['Points']) > 2:
            curve.append(temp)
    curve = {i['Label']: np.asarray([[j[0], j[1]] for j in i['Points']]) for i in curve}
    landmark_ = json_data['Models']['LandMarkListModel']['Points'][0]['LabelList']
    # 将landmark去除第三个维度
    landmark = np.zeros((landmark_num, 2))
    for i in landmark_:
        landmark[i['Label']-8] = i['Position'][:2]
    return curve, landmark


def get_mask(mask_path):
    """获取标注的mask"""
    # !!!!!!!!! np会改变Image的通道排列顺序，Image，为cwh，np为hwc，一般为chw（cv等等） cv:hwc
    mask_img = Image.open(mask_path)  # Image： c, w, h
    mask_array = np.array(mask_img)  # np 会将shape变为 h, w, c

    # 生成poly_curve 图
    poly_curve = np.zeros_like(mask_array)
    for i in range(197, 210):  # 上颌骨区域
        poly_curve[mask_array == i] = 1
    for i in range(250, 256):  # 下颌骨区域
        poly_curve[mask_array == i] = 2
    return torch.as_tensor(poly_curve)


def make_2d_heatmap(landmark, size, var=5.0, max_value=None):
    """
    生成一个size大小，中心在landmark处的热图
    :param max_value: 热图中心的最大值
    :param var: 生成热图的方差 （不是标准差）
    """
    height, width = size
    landmark = (landmark[1], landmark[0])
    x, y = torch.meshgrid(torch.arange(0, height), torch.arange(0, width), indexing="ij")  # 一个网格有横纵两个坐标
    p = torch.stack([x, y], dim=2)
    from math import pi, sqrt
    inner_factor = -1 / (2 * var)
    outer_factor = 1 / sqrt(var * (2 * pi))
    mean = torch.as_tensor(landmark)
    heatmap = (p - mean).pow(2).sum(dim=-1)
    heatmap = torch.exp(heatmap * inner_factor)

    # heatmap[heatmap == 1] = 5
    # 将heatmap的最大值进行缩放
    if max_value is not None:
        heatmap = heatmap * max_value
    return heatmap


def get_angles(angles_info, img_name):
    angles = [0] * 5
    angle_idx = {j: i for i, j in enumerate(['IFA', 'MNM', 'FMA', 'FS', 'PL'])}
    idx = angles_info['name'].index(img_name)
    for metric in angle_idx:
        angles[angle_idx[metric]] = angles_info[metric][idx]
    return angles
