import json
import os
import random

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from data_utils.init_data import check_data
from data_utils.init_anno import get_anno, get_mask
from transforms import GetROI, MyPad
from torchvision.transforms.functional import resize


class YanMianDataset(Dataset):
    def __init__(self, root: str, transforms=None, data_type: str = 'train', resize=None, ki=-1, k=5, json_path=None,
                 mask_path=None, txt_path=None, num_classes=6):
        assert data_type in ['train', 'val', 'test'], "data_type must be in ['train', 'val', 'test']"
        assert num_classes in [5, 6, 11]
        self.root = root
        self.transforms = transforms
        self.json_list = []
        self.data_type = data_type
        self.num_classes = num_classes
        self.run_env = '/' if '/data/' in os.getcwd() else '\\'

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
            print('use cross validation, this is {}.'.format(ki))
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
        if self.run_env == '\\':
            self.json_list = [i.replace('\\jsons\\', '/jsons/') for i in self.json_list]

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

        # get image, annotation and mask
        img_name = json_data['FileInfo']['Name']
        img_path = os.path.join(img_root, img_name)
        origin_image = Image.open(img_path)
        origin_image = origin_image.convert('L').convert('RGB')

        curve, landmark = get_anno(json_data)
        self.towards_right = towards_right(origin_image, landmark)
        # check data
        check_data(curve, landmark, json_data['Polys'][0]['Shapes'], json_dir, self.data_type)

        # get polygon mask
        mask_path = os.path.join(self.mask_path, json_dir.split('/')[-1].split('.')[0] + '_mask_255.jpg')
        bone_mask = get_mask(mask_path)

        roi_box = roi_boxes[json_dir] if self.data_type == 'test' else None
        target = {'landmark': landmark, 'mask': bone_mask, 'curve': curve, 'data_type': self.data_type,
                  'num_classes': self.num_classes, 'roi_box': roi_box, 'img_name': img_name}
        # transforms
        roi_img, target = self.transforms(origin_image, target)

        # 生成Gt mask
        if self.data_type == 'test':
            origin_image = resize(origin_image, [int(origin_image.size[1] * target['resize_ratio']),
                                                 int(origin_image.size[0] * target['resize_ratio'])])
            return origin_image, roi_img, target
        return roi_img, target

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


def towards_right(img=None, landmarks=None):
    """
    根据标记点位于图像的方位判断是否需要水平翻转
    若有三分之二的landmark位于图像右上则翻转
    """
    num = 0
    nasion = landmarks[5]
    for i in range(5):
        if landmarks[i][0] > nasion[0]:
            num += 1
    if num >= 3:
        return True
    return False


if __name__ == '__main__':
    import transforms as T

    import matplotlib.pyplot as plt
    mydata = YanMianDataset('./datas', data_type='val', num_classes=11, transforms=T.Compose(
        [T.GetROI(border_size=30),
         T.AffineTransform(input_size=(256, 256)),
         T.CreateGTmask(hm_var=40),
         # T.RandomRotation(10, rotate_ratio=0.7, expand_ratio=0.7),
         T.ToTensor(),
         T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
         T.MyPad(size=256)
         ]))
    for i in range(len(mydata)):
        img, target= mydata[i]
        print(i)


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
