import os
import shutil
import time

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
import cv2
import json

import torchvision.transforms.functional
from PIL import Image
from torchvision.transforms.functional import crop, resize, pad, to_tensor, normalize
from train_utils.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from train_utils.vit_seg_modeling import VisionTransformer as ViT_seg
from detec_backbone import resnet50_fpn_backbone, MobileNetV2
from detec_network_files import FasterRCNN, FastRCNNPredictor, AnchorsGenerator

from eva_utils.my_eval import *
from src import VGG16UNet


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def show_img(img, target, title='', save_path=None):
    img = np.array(img) / 255
    mask = target['mask']
    landmark = target['landmark']
    mask_y, mask_x = torch.where(mask != 0)
    for y, x in zip(mask_y, mask_x):
        img[y, x, 0] = img[y, x, 0] * 0.6 + 0.4
    for i in landmark.values():
        cv2.circle(img, i, 2, (1, 0, 0), -1)
    plt.title(title)
    plt.imshow(img)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def create_detec_model(num_classes=2):
    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d)
    model = FasterRCNN(backbone=backbone, num_classes=num_classes, rpn_score_thresh=0.5, min_size=256, max_size=256)
    return model


def main():
    # init cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    init_img = torch.zeros((1, 3, 256, 256), device=device)

    # init detec model
    detec_model = create_detec_model(num_classes=2)
    detec_weight = "../models/model/detec/data6_SGDlr0.02_0.9169/best_model.pth"
    assert os.path.exists(detec_weight), "{} file dose not exist.".format(detec_weight)
    detec_model.load_state_dict(torch.load(detec_weight, map_location=device)["model"])
    detec_model.to(device)
    detec_model.eval()
    detec_model(init_img)

    # init heatmap model
    # model = UNet(in_channels=3, num_classes=classes + 1, base_c=64)
    heatmap_model = VGG16UNet(num_classes=6)
    heatmap_weight = "../models/model/heatmap/data6_vu_b16_ad_var100_max2/lr_0.0008_3.807/best_model.pth"  # 127, 136
    assert os.path.exists(heatmap_weight), f"weights {heatmap_weight} not found."
    heatmap_model.load_state_dict(torch.load(heatmap_weight, map_location='cpu')['model'])
    heatmap_model.to(device)
    heatmap_model.eval()
    heatmap_model(init_img)

    # init seg model
    seg_weight = '../models/model/cross_validation/pl_SGDlr0.02_ers_b32_0.769/1_0.768/best_model.pth'
    # model_poly_curve = UNet(in_channels=3, num_classes=5, base_c=32)
    # model_poly_curve = VGG16UNet(num_classes=5)
    config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
    config_vit.n_classes = 5
    config_vit.n_skip = 3
    if 'R50-ViT-B_16'.find('R50') != -1:
        config_vit.patches.grid = (
            int(256 / 16), int(256 / 16))
    seg_model = ViT_seg(config_vit, img_size=256, num_classes=5)
    seg_model.load_state_dict(torch.load(seg_weight, map_location='cpu')['model'])
    seg_model.to(device)
    seg_model.eval()
    seg_model(init_img)

    # init datas
    mean = [0.2281, 0.2281, 0.2281]
    std = [0.2313, 0.2313, 0.2313]

    # img data and roi box
    labels = ['18', '21', 'clp']
    img_paths = [{i: os.path.join('./datas', i, j)} for i in labels for j in os.listdir(os.path.join('./datas', i))]
    result_pre = {i: {'name': [], 'IFA': [], 'MNM': [], 'FMA': [], 'FPL': [], 'PL': [], 'MML': [], 'FS': []} for i
                  in labels}
    spacing_cm = pd.read_excel('../data_utils/data_information/spacing_values.xlsx').to_dict('list')
    spacing_cm['name'] = [i.split('.')[0] for i in spacing_cm['name']]

    save_path = './result_230425'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with torch.no_grad():  # 关闭梯度计算功能，测试不需要计算梯度
        for i, data in tqdm(enumerate(img_paths)):
            for label, img_p in data.items():
                pass
            name = img_p.split('\\')[-1].split('.')[0]
            print(name)
            if name in [
                        '0112022_Yumei_Wang_20181224130949650',   # 预测结果不好
                        '5010816_lifang_chang_clp',   # 重复图片
                        '1_0031_72',   # clp detection 无结果
                        '0133204_Qingfeng_Dong_20190719140157679',   # 无法计算指标值
                        ]:
                continue

            if name not in spacing_cm['name']:
                spa = 1
            else:
                spa = spacing_cm['spa'][spacing_cm['name'].index(name)] * 10
            origin_img = Image.open(img_p)
            ori_w, ori_h = origin_img.size

            # use detec model to get ROI box
            detec_img = torchvision.transforms.functional.to_tensor(origin_img)
            detec_img = torch.unsqueeze(detec_img, dim=0)
            detec_pre = detec_model(detec_img.to(device))[0]
            assert len(detec_pre['scores'] > 0)
            best_score_index = np.argmax(detec_pre['scores'].to('cpu'))
            pre_box = detec_pre['boxes'][best_score_index].to('cpu').numpy()
            pre_box = [int(i) for i in pre_box]
            left, top, right, bottom = pre_box
            assert left >= 0 and right <= ori_w and top >= 0 and bottom <= ori_h

            height, width = bottom - top, right - left
            origin_img = origin_img.convert('L').convert('RGB')
            # ROI_img = origin_img.crop(pre_box)
            ROI_img = crop(origin_img, top, left, height, width)
            # resize image [256, 256]
            ratio = 256 / width
            if height * ratio > 256:
                ratio = 256 / height

            # origin_img = origin_img.resize([int(ori_w * ratio), int(ori_h * ratio)])
            # ROI_img = ROI_img.resize([int(width * ratio), int(height * ratio)])
            origin_img = resize(origin_img, [int(ori_h * ratio), int(ori_w * ratio)])
            ROI_img = resize(ROI_img, [int(height * ratio), int(width * ratio)])
            pre_box = [int(i * ratio) for i in pre_box]
            # to_tensor ,normalize
            ROI_img = to_tensor(ROI_img)
            ROI_img = normalize(ROI_img, mean, std)
            # padding to [256,256]
            roi_h, roi_w = ROI_img.shape[-2:]
            w_pad, h_pad = 256 - roi_w, 256 - roi_h
            ROI_img = pad(ROI_img, [0, 0, w_pad, h_pad], fill=0)

            # expand batch dimension
            ROI_img = torch.unsqueeze(ROI_img, dim=0)

            # model1 预测六个点
            output = heatmap_model(ROI_img.to(device))
            prediction = output.squeeze().to('cpu')

            # model2 预测poly_curve
            output2 = seg_model(ROI_img.to(device))
            prediction2 = output2['out'].to('cpu')
            prediction = torch.cat((prediction2.squeeze(), prediction), dim=0)

            # 去除padding的部分
            prediction = prediction[:, 0:roi_h, 0:roi_w]

            # 生成预测数据的统一格式的target{'landmark':landmark,'mask':mask}
            pre_ROI_target, not_exist_landmark = create_predict_target(ROI_img, prediction, img_p, deal_pre=True)

            # 将ROI target 转换为原图大小
            pre_target = create_origin_target(pre_ROI_target, pre_box, origin_img.size)

            # plt.imshow(pre_target['mask'], cmap='gray')
            # plt.show()

            if len(not_exist_landmark) > 0:
                print(not_exist_landmark)
                show_img(origin_img, pre_target, title=name)
                continue

            if name in []:
                show_img(origin_img, pre_target, title=name)
                continue

            pre_data, img = calculate_metrics(origin_img, pre_target, not_exist_landmark, is_gt=False, show_img=False,
                                              compute_MML=True, name=name, save_path=save_path + '/' + label,
                                              resize_ratio=ratio, spa=spa)

            if name in ['2010532_xu_jiangfeng', '0119637_Yan_Cai_20220307083302440', '2411097_HE_HONGZHU']:
                for metric in ['IFA', 'MNM', 'FMA', 'PL', 'FS']:
                    show_one_metric(origin_img, pre_target, metric, not_exist_landmark, spa=spa, box=pre_box,
                                    resize_ratio=ratio, save_path=f'{save_path}/{name}/{metric}.png')
            # shutil.copy(img_p, img_p.replace('datas', 'result_1222'))
            result_pre[label]['name'].append(name)
            for key in ['IFA', 'MNM', 'FMA', 'FPL', 'PL', 'MML', 'FS']:
                result_pre[label][key].append(round(float(pre_data[key]), 2))

    # 阳性诊断
    for label in result_pre:
        print(label, '共{}张有效预测图片'.format(len(result_pre[label]['IFA'])))
        print('      num    max     min     mean    std')
        for metric in result_pre[label]:
            if metric != 'name':
                data = result_pre[label][metric].copy()
                while None in data:
                    data.remove(None)
                print('{:<6} {:<5} {:<5}   {:<5}   {:<5}   {:<5}'.format(metric, len(data), max(data), min(data),
                                                                         round(float(np.mean(data)), 2),
                                                                         round(float(np.std(data)), 2)))
    # dice 误差
    with pd.ExcelWriter(save_path + '/result_pre.xlsx') as writer:
        for i in result_pre:
            df = pd.DataFrame(result_pre[i])
            df.to_excel(writer, sheet_name=i, index=False)


if __name__ == '__main__':
    main()
