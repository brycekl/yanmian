import json
import os.path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
import torch
from eva_utils.eval_metric import *
from yanMianDataset import towards_right as compute_towards_right


def get_ground_truth(json_dir, mask_root):
    target = {}
    # load json data
    json_str = open(json_dir, 'r')
    json_data = json.load(json_str)
    json_str.close()

    # get curve, landmark data
    contours_list_curve = json_data['Models']['PolygonModel2']
    curve = []
    # 去除标curve时多标的点
    for contours_list in contours_list_curve:
        if len(contours_list['Points']) > 2:
            curve.append(contours_list)
    landmark = json_data['Models']['LandMarkListModel']['Points'][0]['LabelList']
    # 将landmark转换为int，同时去除第三个维度
    landmark = {i['Label']: np.array(i['Position'], dtype=np.int32)[:2] for i in landmark}

    # get polygon mask
    mask_img = Image.open(mask_root)
    mask_array = np.array(mask_img)
    # 将mask上的像素值转为label
    mask = np.zeros(mask_array.shape, np.uint8)
    mask[mask_array == 4] = 1  # 上颌骨
    mask[mask_array == 5] = 2  # 下颌骨

    # 在mask绘制曲线  3（鼻根曲线），4（前额曲线）
    for idx in range(len(curve)):
        points = curve[idx]['Points']
        label = curve[idx]['Label']
        points_array = np.array(points, dtype=np.int32)[:, :2]
        for i in range(len(points_array) - 1):
            cv2.line(mask, points_array[i], points_array[i + 1], color=label - 3, thickness=6)
    target['mask'] = torch.as_tensor(mask)
    target['landmark'] = landmark

    return target


def create_predict_target(img, prediction, json_dir, towards_right=True, deal_pre: bool = False):
    if isinstance(img, torch.Tensor):
        h, w = img.shape[-2:]
    else:
        w, h = img.size
    poly_curve = functional.softmax(prediction[:5], dim=0)
    # 去除部分预测强度不高的数据
    poly_curve[poly_curve < 0.85] = 0
    poly_curve = np.argmax(poly_curve, axis=0)
    # poly_curve 其实已经可以作为mask,下面对它进行闭运算,腐蚀等处理
    not_exist_landmark = []  # 统计预测不存在的点

    landmark = {}
    # get the result of landmark
    for i in range(5, 11):
        temp = np.array(prediction[i])
        # 先判断不进行操作时,是否存在预测的点,若有,则保存
        y, x = np.where(temp == temp.max())
        if len(x) == 0:
            not_exist_landmark.append(i)
        elif len(x) == 1:
            landmark[i + 3] = [x[0], y[0]]
        elif len(x) > 1:
            print('{} 预测关键点个数大于1'.format(json_dir))
            # 一直对mask进行腐蚀,得到最精确的定位点
            while len(x) > 1:
                temp2 = cv2.erode(temp, np.ones((3, 3)), iterations=1)
                y2, x2 = np.where(temp2 == temp2.max())
                # 如果腐蚀后不存在最大值了，则直接使用上一次的结果
                if len(x2) == 0:
                    break
                # 否则继续腐蚀
                elif len(x2) >= 1:
                    x = x2
                    y = y2
                    temp = temp2
            landmark[i + 3] = [x[0], y[0]]

    towards_right = compute_towards_right(img, landmark)
    if not deal_pre:
        for i in range(1, 5):
            y, x = np.where(poly_curve == i)
            if len(y) == 0:
                not_exist_landmark.append(i)
                continue
        mask = np.array(poly_curve)
    else:
        # 在点定位精度较高的情况下，根据点定位结果优化两个区域和曲线
        nasion = landmark[13]

        mask = np.zeros(prediction.shape[-2:], dtype=np.uint8)
        kernel = np.ones((3, 3), dtype=np.uint8)  # 对每个区域做闭运算，去除缺陷
        for i in range(1, 5):
            temp = np.zeros_like(mask)
            temp[poly_curve == i] = 255
            # 开运算去除瑕疵， 闭运算连通小区域
            # temp = cv2.morphologyEx(temp, cv2.MORPH_OPEN, kernel, iterations=1)
            # temp = cv2.morphologyEx(temp, cv2.MORPH_CLOSE, kernel, iterations=1)
            # 获取ROI区域的坐标点
            y, x = np.where(temp == 255)
            if len(y) == 0:
                not_exist_landmark.append(i)
                continue

            # 220609 只保留预测区域中面积最大的部分
            contours, _ = cv2.findContours(temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            max_area_index = 0
            max_area = 0
            for ind, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area > max_area:
                    max_area = area
                    max_area_index = ind
            # 填充最大面积的区域
            temp = np.zeros_like(mask)
            for ind, contour in enumerate(contours):
                if ind == max_area_index:
                    # thickness = -1代表填充
                    cv2.drawContours(temp, contours, contourIdx=ind, thickness=-1, color=255)

            if i == 3:
                # 去除鼻根为中心的20像素矩形外的结果
                # temp[nasion[1] + 20:, :] = 0
                # temp[:, :nasion[0] - 50] = 0
                # temp[:nasion[1] - 50, :] = 0
                # temp[:, nasion[0] + 50:] = 0
                # tt = np.where(temp == 255)
                # if len(tt[0]) == 0:
                #     temp = np.zeros_like(mask)
                #     temp[poly_curve == i] = 255
                # 判断处理后，斜率是否为0，即为不正确预测
                shift_h, shift_w = np.where(temp == 255)
                # 去除鼻根右边（朝右）或左边多余的点-->朝右只取鼻根点左边的点
                index = shift_w < nasion[0] if towards_right else shift_w > nasion[0]
                if any(index):
                    shift_h = shift_h[index]
                    shift_w = shift_w[index]
                mean_point = [int(np.mean(shift_w)), int(np.mean(shift_h))]
                _, k, _ = get_angle_k_b(mean_point, nasion, h)
                if k == 0:
                    not_exist_landmark.append(i)  # 检查数据
            if i == 4:
                # 去除真实额骨上方的所有最小点，避免误判
                # while 1:
                #     top_x = np.mean(x[np.argwhere(y == y.min())])
                #     index = [[m, n] for m, n in zip(x, y) if
                #              (top_x - 10 < m < top_x + 10 and y.min() - 10 < n < y.min() + 10)]
                #     if len(index) < 30:
                #         for m, n in index:
                #             temp[n, m] = 0
                #         y, x = np.where(temp == 255)
                #     elif len(index) == 0 or len(index) >= 30:
                #         break
                #     if len(x) == 0:  # 此处说明处理后，额骨区域全部为0了，即预测的全部区域都为小区域，所以回到未处理的状态
                #         temp = np.zeros_like(mask)
                #         temp[poly_curve == i] = 255
                #         y, x = np.where(temp == 255)
                #         break
                top_x = x[np.argwhere(y == y.min())]
                if towards_right:
                    temp[nasion[1]:, :] = 0  # 去除鼻根下方，右方，额骨最高点左方上方的额骨预测结果
                    temp[:, nasion[0]:] = 0
                else:
                    temp[nasion[1]:, :] = 0
                    temp[:, :nasion[0]] = 0
                tt = np.where(temp == 255)
                if len(tt[0]) == 0:
                    temp[poly_curve == i] = 255
            mask[temp == 255] = i

    # plt.imshow(mask, cmap='gray')
    # plt.show()
    mask = torch.as_tensor(mask)
    target = {'mask': mask, 'landmark': landmark}
    if len(not_exist_landmark) > 0:
        not_exist_landmark.append(json_dir)
    return target, not_exist_landmark


def create_origin_target(ROI_target, box, origin_size):
    ROI_mask = torch.as_tensor(ROI_target['mask'])
    roi_h, roi_w = ROI_mask.shape[-2], ROI_mask.shape[-1]
    if len(ROI_mask.shape) == 3:
        ROI_mask_ = torch.zeros((roi_h, roi_w))
        for i, mask in enumerate(ROI_mask):
            ROI_mask_[ROI_mask[i] != 0] = i
        ROI_mask = ROI_mask_
    ROI_landmark = ROI_target['landmark']
    left, top, right, bottom = box
    mask = torch.zeros((origin_size[1], origin_size[0]))
    mask[top:roi_h + top, left:roi_w + left].copy_(ROI_mask)
    landmark = {i: [j[0] + left, j[1] + top] for i, j in ROI_landmark.items()}
    target = {'mask': mask, 'landmark': landmark}
    return target


def show_predict(rgb_img, prediction, classes):
    import torch
    img = rgb_img.copy()
    if prediction.shape[0] == 5 or prediction.shape[0] == 11:
        # 两个区域和两条线
        poly_curve = torch.nn.functional.softmax(prediction[:5], dim=0)
        poly_curve[poly_curve < 0.5] = 0  # 去除由于裁剪的重复阴影，同时避免小值出现
        poly_curve = np.argmax(poly_curve, axis=0)
        for j in range(1, 5):
            mask = torch.eq(poly_curve, j)
            img[..., 0][mask] = 200
            img[..., 1][mask] = 100
            img[..., 2][mask] = 200
    if prediction.shape[0] == 6 or prediction.shape[0] == 11:
        # 上唇，下唇，上颌前缘中点， 下颌前缘中点，下巴，鼻根
        for i in range(classes - 6, classes):
            keypoint = prediction[prediction.shape[0] - 6 + i]
            keypoint = np.array(keypoint.to('cpu'))
            h_shifts, w_shifts = np.where(keypoint == keypoint.max())  # np返回 行，列--》对应图片为h， w
            w_shift, h_shift = w_shifts[0], h_shifts[0] + 1
            cv2.circle(img, [w_shift, h_shift], radius=6, color=(0, 255, 0), thickness=-1)  # 点坐标x,y应为w_shift,h_shift
    cv2.putText(img, 'red: gt', [20, 20], cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 0, 0), thickness=1)
    cv2.putText(img, 'green: pre', [20, 35], cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 255, 0), thickness=1)
    plt.imshow(img)
    plt.show()


def show_one_metric(rgb_img, gt, pre, metric: str, not_exist_landmark, show_img: bool = False):
    assert metric in ['IFA', 'MNM', 'FMA', 'PL', 'MML'], "metric must in ['IFA', 'MNM', 'FMA', 'PL', 'MML']"
    landmark_gt = gt['landmark']
    mask_gt = gt['mask']
    landmark_pre = pre['landmark']
    mask_pre = pre['mask']

    towards_right1 = compute_towards_right(rgb_img, landmark_pre)
    towards_right2 = compute_towards_right(rgb_img, landmark_gt)
    assert towards_right1 == towards_right2, '定位偏差过大'

    upper_lip_gt = landmark_gt[8]
    under_lip_gt = landmark_gt[9]
    upper_midpoint_gt = landmark_gt[10]
    under_midpoint_gt = landmark_gt[11]
    chin_gt = landmark_gt[12]
    nasion_gt = landmark_gt[13]
    upper_lip_pre = landmark_pre[8]
    under_lip_pre = landmark_pre[9]
    upper_midpoint_pre = landmark_pre[10]
    under_midpoint_pre = landmark_pre[11]
    chin_pre = landmark_pre[12]
    nasion_pre = landmark_pre[13]
    h_img = mask_gt.shape[0]

    img = np.array(rgb_img)
    if metric == 'IFA':
        # 面——下部角 （IFA）  --->求不好 144 238 144 180 238 180  46 139 87
        data_gt = calculate_IFA(img, mask_gt, 3, not_exist_landmark, nasion_gt, chin_gt, upper_lip_gt, under_lip_gt,
                                towards_right1, color=(160, 0, 0), color_point=(160, 0, 0), color_area=(160, 0, 0))
        data_pre = calculate_IFA(img, mask_pre, 3, not_exist_landmark, nasion_pre, chin_pre, upper_lip_pre,
                                 under_lip_pre, towards_right2, color=(46, 139, 87), color_point=(46, 139, 87),
                                 color_area=(46, 139, 87))
    elif metric == 'MNM':
        # 上颌 10 -鼻根 13 -下颌 11角（MNM角）   ----> 完成
        data_gt = calculate_MNM(img, not_exist_landmark, nasion_gt, upper_midpoint_gt, under_midpoint_gt,
                                color=(160, 0, 0), color_point=(160, 0, 0))
        data_pre = calculate_MNM(img, not_exist_landmark, nasion_pre, upper_midpoint_pre, under_midpoint_pre,
                                 color=(46, 139, 87), color_point=(46, 139, 87))
    elif metric == 'FMA':
        # 面——上颌角（FMA）    -----> 基本完成
        data_gt = calculate_FMA(img, mask_gt, 1, not_exist_landmark, upper_lip_gt, chin_gt, towards_right1,
                                color=(160, 0, 0), color_point=(160, 0, 0), color_area=(160, 0, 0))
        data_pre = calculate_FMA(img, mask_pre, 1, not_exist_landmark, upper_lip_pre, chin_pre, towards_right2,
                                 color=(46, 139, 87), color_point=(46, 139, 87), color_area=(46, 139, 87))
    elif metric == 'PL' or metric == 'MML':
        MML_img = img.copy()
        # 颜面轮廓线（FPL） & 颜面轮廓（PL）距离     -----> 完成
        data_gt, _, big_head_point_gt = calculate_PL(img, mask_gt, 4, not_exist_landmark, under_midpoint_gt,
                                                     nasion_gt, towards_right1, color=(160, 0, 0),
                                                     color_point=(160, 0, 0), color_area=(160, 0, 0))
        data_pre, _, big_head_point_pre = calculate_PL(img, mask_pre, 4, not_exist_landmark, under_midpoint_pre,
                                                       nasion_pre, towards_right2, color=(46, 139, 87),
                                                       color_point=(46, 139, 87), color_area=(46, 139, 87))
        if metric == 'MML':
            img = MML_img
            # 下颌上颌线（MML） & 额前空间（FS）距离     -----> 完成
            data_gt, _ = calculate_MML(img, mask_gt, 4, not_exist_landmark, under_midpoint_gt, upper_midpoint_gt,
                                       big_head_point_gt, towards_right1, color=(160, 0, 0), color_point=(160, 0, 0),
                                       color_area=(160, 0, 0))
            data_pre, _ = calculate_MML(img, mask_pre, 4, not_exist_landmark, under_midpoint_pre, upper_midpoint_pre,
                                        big_head_point_pre, towards_right2, color=(46, 139, 87),
                                        color_point=(46, 139, 87), color_area=(46, 139, 87))
    if show_img:
        # 添加掩膜  °： '\u00B0'
        img[img.shape[0] - 110:img.shape[0] - 30, :250, :] = 0
        cv2.putText(img, 'GT: : ' + str(round(data_gt, 2)), [20, img.shape[0] - 80], cv2.FONT_HERSHEY_COMPLEX, 1.0,
                    (160, 0, 0), 2)
        cv2.putText(img, 'Pre: : ' + str(round(data_pre, 2)), [20, img.shape[0] - 45], cv2.FONT_HERSHEY_COMPLEX, 1.0,
                    (46, 139, 87), 2)
        plt.imshow(img)
        plt.title(metric)
        plt.show()


def calculate_metrics(rgb_img, gt, not_exist_landmark, is_gt: bool = True, show_img: bool = False, resize_ratio=1,
                      spa=1, compute_MML: bool = True, name=None, save_path=None, put_text=True):
    landmark = gt['landmark']
    mask = gt['mask']
    img = np.array(rgb_img)

    towards_right1 = compute_towards_right(Image.fromarray(img), landmark)  # 标签是否其中在图像右侧
    # 虚化预测结果
    pre_mask = np.zeros_like(img)
    for j in range(1, 5):
        mask_ = torch.where(mask == j)
        pre_mask[..., 0][mask_] = 255
        pre_mask[..., 1][mask_] = 0
        pre_mask[..., 2][mask_] = 0
    # for i in range(8, 14):
    #     cv2.circle(img, landmark[i], radius=2, color=(0, 200, 0), thickness=-1)
    a = pre_mask[:, :, 0]
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if a[i][j] != 0:
                for z in range(3):
                    img[i][j][z] = img[i][j][z] * 0.7 + pre_mask[i][j][z] * 0.3
    upper_lip = landmark[8]
    under_lip = landmark[9]
    upper_midpoint = landmark[10]
    under_midpoint = landmark[11]
    chin = landmark[12]
    nasion = landmark[13]
    h_img = mask.shape[0]

    # 面——下部角 （IFA）  --->求不好
    angle_IFA = calculate_IFA(img, mask, 3, not_exist_landmark, nasion, chin, upper_lip, under_lip, towards_right1,
                              color=(173, 255, 47))
    # 上颌 10 -鼻根 13 -下颌 11角（MNM角）   ----> 完成
    angle_MNM = calculate_MNM(img, not_exist_landmark, nasion, upper_midpoint, under_midpoint, color=(255, 215, 0))

    # 面——上颌角（FMA）    -----> 基本完成
    angle_FMA = calculate_FMA(img, mask, 1, not_exist_landmark, upper_lip, chin, towards_right1, color=(255, 106, 106))

    # 颜面轮廓线（FPL） & 颜面轮廓（PL）距离     -----> 完成
    PL, FPL, head_point = calculate_PL(img, mask, 4, not_exist_landmark, under_midpoint, nasion, towards_right1,
                                       color=(0, 191, 255))
    # 颜面轮廓线（MML） & 颜面轮廓（FS）距离     -----> 完成
    if compute_MML:
        FS, MML = calculate_MML(img, mask, 4, not_exist_landmark, under_midpoint, upper_midpoint, head_point,
                                towards_right1, color=(155, 48, 255))
    else:
        FS, MML = -1, 0
    data = {'IFA': angle_IFA, 'MNM': angle_MNM, 'FMA': angle_FMA, 'PL': (PL/resize_ratio)*spa, 'FPL': FPL, 'FS': (FS/resize_ratio)*spa, 'MML': MML}

    # color = (255, 215, 0)
    # color_point = (255, 215, 0)
    # img_ = rgb_img.copy()
    # rgb_img = img_.copy()
    # rgb_img[rgb_img.shape[0] - 52:rgb_img.shape[0] - 15, :240, :] = 0
    # cv2.putText(rgb_img, 'IFA: ' + str(round(angle_IFA, 2)), [20, rgb_img.shape[0] - 23], cv2.FONT_HERSHEY_COMPLEX, 1,
    #             (255, 215, 0), 2)
    # cv2.putText(rgb_img, 'o', [190, rgb_img.shape[0] - 38], cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    #             (255, 215, 0), 2)
    # from torchvision.transforms.functional import to_pil_image
    # t_ = to_pil_image(rgb_img)
    # t_.save('./test.jpg')

    if put_text:
        # 添加掩膜  °： '\u00B0'
        img[img.shape[0] - 215:img.shape[0] - 30, :250, :] = 0
        cv2.putText(img, 'IFA: ' + str(round(angle_IFA, 2)), [20, img.shape[0] - 180], cv2.FONT_HERSHEY_COMPLEX, 1.0,
                    (173, 255, 47), 2)
        ind = np.where(img[img.shape[0]-200:img.shape[0]-180, :250, 0] != 0)[1].max()
        cv2.putText(img, 'o', [ind+5, img.shape[0] - 192], cv2.FONT_HERSHEY_COMPLEX, 0.5,
                    (173, 255, 47), 2)
        cv2.putText(img, 'MNM: ' + str(round(angle_MNM, 2)), [20, img.shape[0] - 145], cv2.FONT_HERSHEY_COMPLEX, 1.0,
                    (255, 215, 0), 2)
        ind = np.where(img[img.shape[0]-165:img.shape[0]-145, :250, 0] != 0)[1].max()
        cv2.putText(img, 'o', [ind+5, img.shape[0] - 157], cv2.FONT_HERSHEY_COMPLEX, 0.5,
                    (255, 215, 0), 2)
        cv2.putText(img, 'FMA: ' + str(round(angle_FMA, 2)), [20, img.shape[0] - 110], cv2.FONT_HERSHEY_COMPLEX, 1.0,
                    (255, 106, 106), 2)
        ind = np.where(img[img.shape[0]-130:img.shape[0]-110, :250, 0] != 0)[1].max()
        cv2.putText(img, 'o', [ind+5, img.shape[0] - 122], cv2.FONT_HERSHEY_COMPLEX, 0.5,
                    (255, 106, 106), 2)
        mm_yes = 'mm' if spa != 1 else ''
        cv2.putText(img, 'PL: ' + str(round(data['PL'], 2)) + mm_yes, [20, img.shape[0] - 75], cv2.FONT_HERSHEY_COMPLEX,
                    1.0, (0, 191, 255), 2)
        cv2.putText(img, 'FS: ' + str(round(data['FS'], 2)) + mm_yes, [20, img.shape[0] - 40], cv2.FONT_HERSHEY_COMPLEX,
                    1.0, (155, 48, 255), 2)
    img = Image.fromarray(img)
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # plt.imsave(os.path.join(save_path, name + '_pre.png'), img)
        img.save(os.path.join(save_path, name + '.png'))
    if show_img:
        plt.imshow(img)
        plt.title(name)
        plt.show()
        plt.close()
    return data, img
