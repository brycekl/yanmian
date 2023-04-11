import numpy as np
from PIL import Image
import cv2
import torch


def get_anno(json_data):
    temp_curve = json_data['Models']['PolygonModel2']  # list   [index]['Points']
    curve = []
    # 去除标curve时多标的点
    for temp in temp_curve:
        if len(temp['Points']) > 2:
            curve.append(temp)
    curve = {i['Label']: [[j[0], j[1]] for j in i['Points']] for i in curve}

    landmark = json_data['Models']['LandMarkListModel']['Points'][0]['LabelList']
    # 将landmark去除第三个维度
    landmark = {i['Label']: np.array(i['Position'])[:2] for i in landmark}
    return curve, landmark


def get_mask(mask_path, curve):
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
    for label, points in curve.items():
        points_array = np.array(points, dtype=np.int32)
        for j in range(len(points_array) - 1):
            cv2.line(poly_curve, points_array[j], points_array[j + 1], color=label - 3, thickness=2)
    return poly_curve


def create_GTmask(target, hm_max=8):
    """生成训练的GT mask"""
    landmark = target['landmark']
    num_classes = target['num_classes']
    # 生成mask, landmark的误差在int()处
    landmark = {i: [int(landmark[i][0]), int(landmark[i][1])] for i in landmark}
    mask = torch.zeros(num_classes, *target['mask'].shape, dtype=torch.float)
    # 根据landmark 绘制高斯热图 （进行点分割）
    # heatmap 维度为 c,h,w 因为ToTensor会将Image(c.w,h)也变为(c,h,w)
    if num_classes == 6 or num_classes == 11:
        for label in landmark:
            point = landmark[label]
            temp_heatmap = make_2d_heatmap(point, target['mask'].shape, var=target['hm_var'], max_value=hm_max)
            mask[label - 8] = temp_heatmap
    # todo 优化，poly的信息可以集中在一个通道里，求loss时用one hot分离
    if num_classes == 5 or num_classes == 11:
        num_poly_mask = num_classes - 5
        poly = np.array(target['mask'])
        for label in range(1, 5):
            # 这个写法复杂度有点高可能
            mask[num_poly_mask + label][poly == label] = 1
            mask[num_poly_mask + label][poly == 255] = 255
        mask[num_poly_mask][poly == 0] = 1
        mask[num_poly_mask][poly == 255] = 255

    # 将mask中，landmark左侧的target置为0
    # for i in range(6):
    #     y, x = np.where(mask[i] == mask[i].max())
    #     mask[i, :, :x[0]] = 0
    target['mask'] = mask
    return target


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
