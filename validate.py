import os
import time

import numpy as np
import pandas as pd
import yaml
from yanMianDataset import YanMianDataset
import transforms as T
from eva_utils.my_eval import *
from eva_utils import analyze
import matplotlib.pyplot as plt
from tqdm import tqdm
from train_utils.init_model_utils import create_model
from train_utils.dice_coefficient_loss import multiclass_dice_coeff
from torch.nn.functional import softmax


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def show_img(img, pre_target, target, title='', save_path=None):
    img = np.array(img) / 255
    mask = np.argmax(target['mask'], 0) if len(target['mask'].size()) == 3 else target['mask']
    landmark = target['landmark']
    landmark = {i: [int(landmark[i][0]+0.5), int(landmark[i][1]+0.5)] for i in landmark}
    for x_tick in range(mask.shape[0]):
        for y_tick in range(mask.shape[1]):
            if mask[x_tick][y_tick].item() != 0:
                img[x_tick][y_tick][0] = img[x_tick][y_tick][0] * 0.5 + 0.5
    for i in landmark.values():
        cv2.circle(img, i, 2, (1, 0, 0), -1)

    pre_mask = pre_target['mask']
    pre_landmark = pre_target['landmark']
    for x_tick in range(pre_mask.shape[0]):
        for y_tick in range(pre_mask.shape[1]):
            if pre_mask[x_tick][y_tick].item() != 0:
                img[x_tick][y_tick][1] = img[x_tick][y_tick][1] * 0.5 + 0.4
    for i in pre_landmark.values():
        cv2.circle(img, i, 2, (0, 1, 0), -1)

    plt.title(title)
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.imsave(os.path.join(save_path, title + '.png'), img)
        plt.close()
    else:
        plt.imshow(img)
        plt.show()


def main():
    model_path = 'models/vgg16unet/vgg16unet_var40_3.580_w5_0.768'
    save_root = model_path.replace('models', 'results')

    # basic messages
    name_index = {8: 'upper_lip', 9: 'under_lip', 10: 'upper_midpoint', 11: 'under_midpoint', 12: 'chin', 13: 'nasion'}
    spacing_cm = pd.read_excel('./data_utils/data_information/spacing_values.xlsx')
    spacing_cm = spacing_cm.to_dict('list')
    spacing_cm['name'] = [i.split('.')[0] for i in spacing_cm['name']]
    with open(os.path.join(model_path, 'config.yml'), 'r') as f:
        config = yaml.load(f.read(), Loader=yaml.Loader)
    num_classes1, num_classes2 = config.num_classes, config.num_classes2 if 'num_classes2' in config else 0
    num_classes = num_classes1 + num_classes2

    """ init data """
    data_transform = T.Compose([
        T.GetROI(border_size=30), T.RandomResize(256, 256, shrink_ratio=0), T.ToTensor(),
        T.Normalize(mean=(0.2281, 0.2281, 0.2281), std=(0.2313, 0.2313, 0.2313)), T.MyPad(size=256)])
    test_data = YanMianDataset('./datas', transforms=data_transform, data_type='test', resize=[256, 256],
                               num_classes=5, txt_path='./data_utils/test.txt')
    print(len(test_data))

    ''' init cuda '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    init_img = torch.zeros((1, config.input_channels, *config.input_size), device=device)

    # load model
    model = create_model(model_name=config.model_name, num_classes_1=num_classes1,
                         num_classes_2=num_classes2, input_size=config.input_size)
    model.load_state_dict(torch.load(os.path.join(model_path, 'model.pth'), map_location=device)['model'])
    model.to(device).eval()

    # if model 1poly model
    model2_path = ''
    if num_classes != 11 and model2_path:
        with open(os.path.join(model2_path, 'config.yml'), 'r') as f:
            config2 = yaml.load(f.read(), Loader=yaml.Loader)
        model2 = create_model(model_name=config2.model_name, num_classes=config2.num_classes, input_size=config2.input_size)
        model2.load_state_dict(torch.load(os.path.join(model2_path, 'model.pth'), map_location=device)['model'])
        model2.to(device).eval()

    # begin to test datas
    with torch.no_grad():  # 关闭梯度计算功能，测试不需要计算梯度
        # 初始化需要保存的结果数据
        result_pre = {'name': [], 'IFA': [], 'MNM': [], 'FMA': [], 'FPL': [], 'PL': [], 'MML': [], 'FS': []}
        result_gt = {'name': [], 'IFA': [], 'MNM': [], 'FMA': [], 'FPL': [], 'PL': [], 'MML': [], 'FS': []}
        mse = {'name': [], }
        for i in range(8, 14):
            mse[name_index[i]] = []
        dices = {i: [] for i in ['name', 'background', 'up', 'down', 'nasion', 'skin']}

        # 生成预测图
        for index in tqdm(range(len(test_data))):
            json_dir = test_data.json_list[index]
            img_name = json_dir.split('/')[-1].split('_jpg')[0].split('_JPG')[0]

            assert img_name in spacing_cm['name'], img_name + ' not in spacing_cm'
            sp_cm = spacing_cm['spa'][spacing_cm['name'].index(img_name)] * 10  # * 10为mm，否则为cm

            original_img, pre_img, target = test_data[index]
            show_roi_img, roi_box, resize_ratio = target['show_roi_img'], target['roi_box'], target['resize_ratio']
            towards_right = test_data.towards_right
            # expand batch dimension
            pre_img = torch.unsqueeze(pre_img, dim=0)

            # predict image
            prediction = model(pre_img.to(device)).to('cpu')
            if num_classes != 11 and model2_path:
                prediction2 = model2(pre_img.to(device)).to('cpu')
                prediction = torch.cat((prediction, prediction2), 1)
            prediction = prediction.squeeze()

            # 去除pad的数据
            raw_roi_w, raw_roi_h = show_roi_img.size
            target['mask'] = target['mask'][:, :raw_roi_h, :raw_roi_w]
            prediction = prediction[:, :raw_roi_h, :raw_roi_w]

            # 保存预测的热力图结果
            if not os.path.exists(os.path.join(save_root, 'pre_heatmap')):
                os.makedirs(os.path.join(save_root, 'pre_heatmap'))
            for pre_ind, pre_channel in enumerate(prediction):
                plt.imsave(os.path.join(save_root, 'pre_heatmap', img_name + '_' + str(pre_ind) + '.png'), pre_channel)

            # 计算预测数据的landmark 的 mse误差
            mse['name'].append(img_name)
            for i, data in enumerate(prediction[:6]):
                y, x = np.where(data == data.max())
                point = target['landmark'][i + 8]  # label=i+8
                error = round(math.sqrt(math.pow(x[0] - point[0], 2) + math.pow(y[0] - point[1], 2)), 3)
                mse[name_index[i + 8]].append(error*sp_cm)

            # 计算预测的dice
            dice = multiclass_dice_coeff(softmax(prediction[6:], 0).unsqueeze(0), target['mask'].unsqueeze(0))
            for i, (k, v) in enumerate(dices.items()):
                if k == 'name':
                    dices['name'].append(img_name)
                else:
                    dices[k].append(float(dice[i-1]))
            # print(f'dice:{dice:.3f}')

            # 生成预测数据的统一格式的target{'landmark':landmark,'mask':mask}
            pre_ROI_target, not_exist_landmark = create_predict_target(show_roi_img, prediction, json_dir,
                                                                       towards_right=towards_right, deal_pre=True)

            # 将ROI target 转换为原图大小
            gt_target = create_origin_target(target, roi_box, original_img.size)
            pre_target = create_origin_target(pre_ROI_target, roi_box, original_img.size)
            show_img(show_roi_img, pre_ROI_target, target, title=img_name, save_path=os.path.join(save_root, 'pre_results'))

            # 分指标展示'IFA', 'MNM', 'FMA', 'PL', 'MML'
            # for metric in ['IFA', 'MNM', 'FMA', 'PL', 'MML']:
            #     show_one_metric(original_img, target, pre_target, metric, not_exist_landmark, show_img=True)
            # 计算颜面的各个指标
            pre_data, pre_img = calculate_metrics(original_img, pre_target, not_exist_landmark, is_gt=False,
                                                  resize_ratio=resize_ratio,  show_img=False, compute_MML=True,
                                                  spa=sp_cm, name=img_name, save_path=save_root + '/PRE')
            gt_data, gt_img = calculate_metrics(original_img, gt_target, not_exist_landmark=[], show_img=False,
                                                resize_ratio=resize_ratio, compute_MML=True, spa=sp_cm,
                                                name=img_name, save_path=save_root + '/GT')
            # save results and predicted images
            result_pre['name'].append(img_name)
            result_gt['name'].append(img_name)
            for key in ['IFA', 'MNM', 'FMA', 'FPL', 'PL', 'MML', 'FS']:
                errorkey = ''
                result_pre[key].append(pre_data[key])
                result_gt[key].append(gt_data[key])
                if pre_data[key] != 'not' and gt_data[key] != 'not':
                    error = round(gt_data[key] - pre_data[key], 3)
                if key == 'IFA' and (error > 13.41 or error < -11.75):
                    errorkey = 'IFA'
                if key == 'MNM' and (error > 4.41 or error < -3.45):
                    errorkey = 'MNM'
                if key == 'FMA' and (error > 10.98 or error < -10.75):
                    errorkey = 'FMA'
                if key == 'PL' and (error > 0.8 or error < -0.7):
                    errorkey = 'PL'
                if key == 'FS' and (error > 2.0 or error < -2.0):
                    errorkey = 'FS'
                if len(errorkey) > 0:
                    save_error_path = os.path.join(save_root + '/error', errorkey)
                    if not os.path.exists(save_error_path):
                        os.makedirs(save_error_path)
                    pre_img.save(os.path.join(save_error_path, img_name + '_pre_' + str(float(error)) + '.png'))
                    gt_img.save(os.path.join(save_error_path, img_name + '_gt_' + str(float(error)) + '.png'))

    # 评估 mse误差   var100_mse_Rightcrop最佳
    df = pd.DataFrame(mse)
    df.to_excel(save_root + '/mse_mm.xlsx')

    # dice 误差
    mean_dice = [np.asarray(j).mean() for i, j in dices.items() if i != 'name']
    df = pd.DataFrame(dices)
    df.to_excel(save_root + '/dices.xlsx')

    # save
    df_gt = pd.DataFrame(result_gt)
    df_gt.to_excel(save_root + '/result_gt.xlsx', index=True)
    df_pre = pd.DataFrame(result_pre)
    df_pre.to_excel(save_root + '/result_pre.xlsx', index=True)

    # 保存个点mse 结果到txt
    result_info = 'MSE:\n'
    for i in mse:
        if i == 'name':
            continue
        dd = np.asarray(mse[i])
        result_info += f'{i}  {dd.mean():.2f}  {dd.std():.2f}\n'
        for j in [0.2, 0.3, 0.4, 0.5]:
            result_info += '{}  {}\n'.format(j, len(np.where(dd < j)[0] / 500))
    # dice 结果
    result_info += f'Dice: {[round(dice, 2) for dice in mean_dice]}\n\n'
    # 各个指标误差
    for i in ['IFA', 'MNM', 'FMA', 'PL', 'FS']:
        pre = result_pre[i]
        gt = result_gt[i]
        result_info += f'{i} 没有预测：{pre.count(-1)}\n'
        # python 数组可以用count 不可以用 Ture or False作为索引， numpy 数组可以用True or False作为索引，不能用count
        # 去除没有预测的数据，再计算误差
        if pre.count(-1) > 0:
            temp = pre
            temp_gt = gt
            pre = []
            gt = []
            for i in range(len(temp)):
                if temp[i] != -1:
                    pre.append(temp[i])
                    gt.append(temp_gt[i])
        error = []
        for m, n in zip(pre, gt):
            error.append(abs(m - n))
        result_info += f'error mean: {np.mean(error)} \nerror std: {np.std(error)}\n'
    for i in ['FPL', 'MML']:
        result_info += f'{i}\n'
        result_info += f"not gt: {result_gt[i].count('not')},   pre: {result_pre[i].count('not')}\n"
        result_info += f"阳性 1  gt: {result_gt[i].count(1)},   pre: {result_pre[i].count(1)}\n"
        result_info += f"阴性 -1  gt: {result_gt[i].count(-1)},   pre: {result_pre[i].count(-1)}\n"
        result_info += f"0  gt: {result_gt[i].count(0)},   pre: {result_pre[i].count(0)}\n"
    result_info += '\nICC\n'
    # 计算icc
    for metric in result_pre:
        if metric in ['IFA', 'MNM', 'FMA', 'FS', 'PL']:
            icc_1, icc_k = analyze.metric_icc(result_pre, result_gt, metric)
            result_info += '{}: icc_1: {:.3f}   icc_k: {:.3f}\n'.format(metric, icc_1, icc_k)
            print('{}: icc_1: {:.3f}   icc_k: {:.3f}'.format(metric, icc_1, icc_k))
            analyze.metric_BA(result_pre, result_gt, metric, save_path=save_root + '/BA/')

    # 保存结果到txt里
    with open(os.path.join(save_root, 'result.txt'), "a") as f:
        f.write(result_info)


if __name__ == '__main__':
    main()
