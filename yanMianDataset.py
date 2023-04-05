import json
import os
import random

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from data_utils.init_data import check_data
from transforms import GetROI, MyPad
from torchvision.transforms.functional import resize


class YanMianDataset(Dataset):
    def __init__(self, root: str, transforms=None, data_type: str = 'train', resize=None, ki=-1, k=5, json_path=None,
                 mask_path=None, txt_path=None, num_classes=6, var=1):
        assert data_type in ['train', 'val', 'test'], "data_type must be in ['train', 'val', 'test']"
        assert num_classes in [4, 6, 10]
        self.root = root
        self.transforms = transforms
        self.resize = resize
        self.json_list = []
        self.var = var
        self.data_type = data_type
        self.num_classes = num_classes
        self.run_env = '/' if '/data/lk' in os.getcwd() else '\\'

        # read txt file and save all json file list (train/val/test)
        if json_path is None:
            json_path = os.path.join(self.root, 'jsons')
        if txt_path is None:
            txt_path = os.path.join(self.root.replace('datas', 'data_utils'), data_type + '.txt')
        if mask_path is not None:
            self.mask_path = mask_path
        else:
            self.mask_path = None
        assert os.path.exists(txt_path), 'not found {} file'.format(data_type + '.txt')

        with open(txt_path) as read:
            txt_path = [line.strip() for line in read.readlines() if len(line.strip()) > 0]
        if ki != -1:
            # 使用k折交叉验证
            assert data_type in ['train', 'val'], 'test can not use cross validation'
            random.seed(1)
            random.shuffle(txt_path)
            length = len(txt_path) // k
            if data_type == 'val':
                txt_path_ = txt_path[length * ki:length * (ki + 1)]
            else:
                txt_path_ = txt_path[:length * ki] + txt_path[length * (ki + 1):]
            self.json_list = [os.path.join(json_path, i) for i in txt_path_]
        else:
            self.json_list = [os.path.join(json_path, i) for i in txt_path]

        # check file
        assert len(self.json_list) > 0, 'in "{}" file does not find any information'.format(data_type + '.txt')
        for json_dir in self.json_list:
            assert os.path.exists(json_dir), 'not found "{}" file'.format(json_dir)

    def __len__(self):
        return len(self.json_list)

    def __getitem__(self, index):
        img_root = os.path.join(self.root, 'images')
        if self.mask_path is None:
            self.mask_path = os.path.join(self.root, 'masks')

        # load json data
        json_dir = self.json_list[index]
        json_str = open(json_dir, 'r', encoding='utf-8')
        json_data = json.load(json_str)
        json_str.close()

        # get detec roi box
        if self.data_type == 'test':
            with open('data_utils/detec_roi_json/celiang500_boxes.json') as f:
                roi_boxes = json.load(f)

        # get image
        img_name = json_data['FileInfo']['Name']
        img_path = os.path.join(img_root, img_name)
        origin_image = Image.open(img_path)
        # 转换为灰度图，再变为三通道
        origin_image = origin_image.convert('L')
        origin_image = origin_image.convert('RGB')

        target = {}
        # get curve, landmark data
        temp_curve = json_data['Models']['PolygonModel2']  # list   [index]['Points']
        curve = []
        # 去除标curve时多标的点
        for temp in temp_curve:
            if len(temp['Points']) > 2:
                curve.append(temp)
        landmark = json_data['Models']['LandMarkListModel']['Points'][0]['LabelList']
        # 将landmark去除第三个维度
        landmark = {i['Label']: np.array(i['Position'])[:2] for i in landmark}
        self.towards_right = towards_right(origin_image, landmark)
        poly_points = json_data['Polys'][0]['Shapes']
        # get polygon mask
        mask_path = os.path.join(self.mask_path, json_dir.split(self.run_env)[-1].split('.')[0] + '_mask_255.jpg')
        # !!!!!!!!! np会改变Image的通道排列顺序，Image，为cwh，np为hwc，一般为chw（cv等等） cv:hwc
        mask_img = Image.open(mask_path)  # Image： c, w, h
        mask_array = np.array(mask_img)  # np 会将shape变为 h, w, c

        # check data
        check_data(curve, landmark, poly_points, json_dir, self.data_type)
        curve = {i['Label']: [[j[0], j[1]] for j in i['Points']] for i in curve}

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
        poly_curve = torch.as_tensor(poly_curve)
        # 得到标注的ROI区域图像-->单纯裁剪
        roi_box = roi_boxes[json_dir] if self.data_type == 'test' else None
        raw_roi_img, poly_curve, landmark, curve, roi_box = GetROI(border_size=30)(origin_image, {'mask': poly_curve, 'landmark': landmark,
                                                                        'curve': curve, 'roi_box': roi_box})

        # Image，和tensor通道组织方式不一样，但是还可以使用同一个transform是因为它内部根据类型做了处理
        if self.transforms is not None:
            roi_img, target = self.transforms(raw_roi_img, {'landmark': landmark, 'mask': poly_curve, 'curve': curve})
            roi_box = [int(i * target['resize_ratio']) for i in roi_box]
            origin_image = resize(origin_image, [int(origin_image.size[1] * target['resize_ratio']),
                                                 int(origin_image.size[0] * target['resize_ratio'])])

        # 生成mask, landmark的误差在int()处
        landmark = {i: [int(target['landmark'][i][0]), int(target['landmark'][i][1])] for i in target['landmark']}
        num_mask = self.num_classes if self.num_classes == 6 else self.num_classes + 1
        mask = torch.zeros(num_mask, *roi_img.shape[-2:], dtype=torch.float)
        # 根据landmark 绘制高斯热图 （进行点分割）
        # heatmap 维度为 c,h,w 因为ToTensor会将Image(c.w,h)也变为(c,h,w)
        if self.num_classes == 6 or self.num_classes == 10:
            for label in landmark:
                point = landmark[label]
                temp_heatmap = make_2d_heatmap(point, roi_img.shape[-2:], var=self.var, max_value=8)
                mask[label - 8] = temp_heatmap
        # todo 优化，poly的信息可以集中在一个通道里，求loss时用one hot分离
        if self.num_classes == 4 or self.num_classes == 10:
            num_poly_mask = self.num_classes - 4
            poly = np.array(target['mask'])
            for label in range(1, 5):
                label_mask = poly == label
                mask[num_poly_mask + label][label_mask] = 1
            mask[num_poly_mask][poly == 0] = 1
        # 将mask中，landmark左侧的target置为0
        # for i in range(6):
        #     y, x = np.where(mask[i] == mask[i].max())
        #     mask[i, :, :x[0]] = 0
        pre_img, mask, landmark = MyPad(256)(roi_img, mask, landmark, self.data_type)
        # img, mask = RandomRotation(20)(img, mask)

        target['mask'] = mask
        target['landmark'] = landmark
        target['img_name'] = img_name

        if self.data_type == 'test':
            return origin_image, roi_img, pre_img, target, roi_box, target['resize_ratio']
        return pre_img, target

    @staticmethod
    def collate_fn(batch):  # 如何取样本，实现自定义的batch输出
        images, targets = list(zip(*batch))  # batch里每个元素表示一张图片和一个gt
        batched_imgs = cat_list(images, fill_value=0)  # 统一batch的图片大小
        mask = [i['mask'] for i in targets]
        batched_targets = {'landmark': [i['landmark'] for i in targets]}
        batched_targets['img_name'] = [i['img_name'] for i in targets]
        batched_targets['mask'] = cat_list(mask, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))  # 获取每个维度的最大值
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)  # batch, c, h, w
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


def make_2d_heatmap(landmark, size, max_value=None, var=5.0):
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


def towards_right(img, landmarks):
    """
    根据标记点位于图像的方位判断是否需要水平翻转
    若有三分之二的landmark位于图像右上则翻转
    """
    num = 0
    nasion = landmarks[13]
    for i in range(8, 13):
        if landmarks[i][0] > nasion[0]:
            num += 1
    if num >= 3:
        return True
    return False

# from transforms import RightCrop
# d = os.getcwd()
# mydata = YanMianDataset(d, data_type='test', resize=[320,320])  # , transforms=RightCrop(2/3),resize=[256,384]
# # a,b = mydata[0]
# # c =1
# for i in range(len(mydata)):
#     a,b = mydata[i]
#     print(i)


# train data 1542
# val data 330
# test data 330

# data 1
# 试标 22张不能用
# 1 curve: 5, 5 landmark: 3, 上颌骨（下颌骨）未被标注（无label）:7, 存在曲线未被标注（无label）:7
# data 2
# IMG_20201021_2_55_jpg_Label.json 只标了一条线，且一条线只有一个点
# 0135877_Mengyan_Tang_20200917091142414_jpg_Label.json  未标注曲线
# 1 curve: 3, 5 landmark:6, 上颌骨（下颌骨）未被标注（无label）:17, 存在曲线未被标注（无label）:1
# data 3
# 1 curve: 1, 5 landmark: 5, 上颌骨（下颌骨）未被标注（无label）:2, 存在曲线未被标注（无label）:0
# data 4
# 1 curve: 1, 5 landmark: 5, 上颌骨（下颌骨）未被标注（无label）:9, 存在曲线未被标注（无label）:0
# data 5
# 1 curve: 5, 5 landmark:0, 上颌骨（下颌骨）未被标注（无label）:14, 存在曲线未被标注（无label）:0
# data 6
# 1 curve: 2, 5 landmark:0, 上颌骨（下颌骨）未被标注（无label）:12, 存在曲线未被标注（无label）:0
# 0117667_Yinying_Chen_20210526131902731_jpg_Label
