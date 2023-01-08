import os
import time

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

import torchvision.transforms.functional
from PIL import Image
from torchvision.transforms.functional import crop, pad, to_tensor, normalize
from train_utils.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from train_utils.vit_seg_modeling import VisionTransformer as ViT_seg
from detec_backbone import resnet50_fpn_backbone, MobileNetV2
from detec_network_files import FasterRCNN, FastRCNNPredictor, AnchorsGenerator

from eva_utils.my_eval import *
from src import VGG16UNet


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def show_img(img, target, title=''):
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
    plt.show()


def create_detec_model(num_classes=2):
    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d)
    model = FasterRCNN(backbone=backbone, num_classes=num_classes, rpn_score_thresh=0.5, min_size=256, max_size=256)
    return model


def main():
    # init cuda
    run_env = '/' if '/data/lk' in os.getcwd() else '\\'
    save_path = './result/predict_500'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    init_img = torch.zeros((1, 3, 256, 256), device=device)

    # init detec model
    detec_model = create_detec_model(num_classes=2)
    # load train weights
    detec_weight = "./model/detec/data6_SGDlr0.02_0.9169/best_model.pth"
    assert os.path.exists(detec_weight), "{} file dose not exist.".format(detec_weight)
    detec_model.load_state_dict(torch.load(detec_weight, map_location=device)["model"])
    detec_model.to(device)
    detec_model.eval()
    detec_model(init_img)

    # init heatmap model
    # model = UNet(in_channels=3, num_classes=classes + 1, base_c=64)
    heatmap_model = VGG16UNet(num_classes=6)
    heatmap_weight = "./model/heatmap/data6_vu_b16_ad_var100_max2/lr_0.0008_3.807/best_model.pth"  # 127, 136
    assert os.path.exists(heatmap_weight), f"weights {heatmap_weight} not found."
    heatmap_model.load_state_dict(torch.load(heatmap_weight, map_location='cpu')['model'])
    heatmap_model.to(device)
    heatmap_model.eval()
    heatmap_model(init_img)

    # init seg model
    seg_weight = './model/cross_validation/pl_SGDlr0.02_ers_b32_0.769/1_0.768/best_model.pth'
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
    index = {8: 'upper_lip', 9: 'under_lip', 10: 'upper_midpoint', 11: 'under_midpoint',
             12: 'chin', 13: 'nasion'}

    with open('data_utils/test.txt') as read:
        test_paths = [line.strip() for line in read.readlines() if len(line.strip()) > 0]
    img_paths = [os.path.join('./data/images', i.split('_jpg')[0] + '.jpg') for i in test_paths]
    result_pre = {'name': [], 'IFA': [], 'MNM': [], 'FMA': [], 'FPL': [], 'PL': [], 'MML': [], 'FS': []}
    with open('data_utils/detec_roi_json/celiang500_boxes.json') as read:
        json_list = json.load(read)
        json_list = {i.split(run_env)[-1].split('_jpg')[0]: j for i, j in json_list.items()}

    with torch.no_grad():  # 关闭梯度计算功能，测试不需要计算梯度
        for i, img_p in tqdm(enumerate(img_paths)):
            name = img_p.split(run_env)[-1].split('.')[0]
            # print(name)
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
            assert pre_box == json_list[name]
            left, top, right, bottom = pre_box
            assert left >= 0 and right <= ori_w and top >= 0 and bottom <= ori_h

            height, width = bottom - top, right - left
            origin_img = origin_img.convert('L').convert('RGB')
            ROI_img = origin_img.crop(pre_box)
            # ROI_img = crop(origin_img, top, left, height, width)
            # resize image [256, 256]
            # ratioes = [256/width, 256/height]
            # if any([i < 1 for i in ratioes]):
            #     ratio = min(ratioes)
            # else:
            #     ratio = max(ratioes)
            ratio = 256 / width
            if height * ratio > 256:
                ratio = 256 / height

            origin_img = origin_img.resize([int(ori_w * ratio), int(ori_h * ratio)])
            ROI_img = ROI_img.resize([int(width * ratio), int(height * ratio)])
            # origin_img = resize(origin_img, [int(ori_h * ratio), int(ori_w * ratio)])
            # ROI_img = resize(ROI_img, [int(height * ratio), int(width * ratio)])
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
                show_img(origin_img, pre_target)
            # 分指标展示'IFA', 'MNM', 'FMA', 'PL', 'MML'
            # for metric in ['IFA', 'MNM', 'FMA', 'PL', 'MML']:
            #     show_one_metric(original_img, target, pre_target, metric, not_exist_landmark, show_img=True)
            # 计算颜面的各个指标
            pre_data, img = calculate_metrics(origin_img, pre_target, not_exist_landmark, is_gt=False, show_img=False,
                                              compute_MML=True, name=name, save_path=save_path + '/pre_images',
                                              resize_ratio=ratio)

            result_pre['name'].append(name)
            for key in ['IFA', 'MNM', 'FMA', 'FPL', 'PL', 'MML', 'FS']:
                result_pre[key].append(float(pre_data[key]))

    from eva_utils import analyze
    df = pd.DataFrame(result_pre)
    df.to_excel(save_path + '/result_pre_.xlsx')


if __name__ == '__main__':
    main()
