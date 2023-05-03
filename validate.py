import os
import shutil
import time
import pandas as pd
from yanMianDataset import YanMianDataset
import transforms as T
from eva_utils.my_eval import *
from eva_utils import analyze
import matplotlib.pyplot as plt
from tqdm import tqdm
from src import VGG16UNet
from train_utils.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from train_utils.vit_seg_modeling import VisionTransformer as ViT_seg
from train_utils.dice_coefficient_loss import build_target, multiclass_dice_coeff


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
    # 运行环境： windows 和 linux
    run_env = "/" if '/data/lk' in os.getcwd() else '\\'
    weights_path = 'models/model/heatmap/data6_vu_b16_ad_var100_max2/lr_0.0008_3.807/best_model.pth'
    test_txt = './data_utils/test.txt'
    save_root = './results/230503'
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(test_txt), f'test.txt {test_txt} not found.'
    
    # roi_mean std
    mean = (0.2281, 0.2281, 0.2281)
    std = (0.2313, 0.2313, 0.2313)
    # from pil image to tensor and normalize
    data_transform = T.Compose([T.Resize([256]), T.ToTensor(), T.Normalize(mean=mean, std=std)])
    test_data = YanMianDataset(os.getcwd(), transforms=data_transform, data_type='test', resize=[256, 256],
                               num_classes=4, txt_path=test_txt)
    print(len(test_data))

    name_index = {8: 'upper_lip', 9: 'under_lip', 10: 'upper_midpoint', 11: 'under_midpoint', 12: 'chin', 13: 'nasion'}

    spacing_cm = pd.read_excel('./data_utils/data_information/spacing_values.xlsx')
    spacing_cm = spacing_cm.to_dict('list')
    spacing_cm['name'] = [i.split('.')[0] for i in spacing_cm['name']]

    # get devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # load heatmap model
    # model = UNet(in_channels=3, num_classes=classes + 1, base_c=64)
    model = VGG16UNet(num_classes=6)

    # load weights
    # 模型融合
    model_merge = False
    if model_merge:
        w_path = ['./model/test_heatmap_lr_b32_ers/3.849_resume_sgdlr0.0005_max10_b8_3.847/best_model.pth',
                  './model/test_heatmap_lr_b32_ers/3.849_resume_sgdlr0.0005_max10_b8_3.841/best_model.pth']
        model_weight = torch.load(weights_path, map_location='cpu')['model']
        for p in w_path:
            temp_weight = torch.load(p, map_location='cpu')['model']
            for weight_name in model_weight:
                model_weight[weight_name] = (model_weight[weight_name] + temp_weight[weight_name]) / 2
        model.load_state_dict(model_weight)
    else:
        model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    model.to(device)
    model.eval()

    # load poly model
    model2 = True
    if model2:  # './model/cross_validation/pl_SGDlr0.02_ers_b32_0.769/1_0.768/best_model.pth'
        weights_poly_curve = './models/model/cross_validation/pl_SGDlr0.02_ers_b32_0.769/1_0.768/best_model.pth'
        # model_poly_curve = UNet(in_channels=3, num_classes=5, base_c=32)
        # model_poly_curve = VGG16UNet(num_classes=5)
        config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
        config_vit.n_classes = 5
        config_vit.n_skip = 3
        if 'R50-ViT-B_16'.find('R50') != -1:
            config_vit.patches.grid = (
                int(256 / 16), int(256 / 16))
        model_poly_curve = ViT_seg(config_vit, img_size=256, num_classes=5)
        # 模型融合
        poly_model_merge = False
        if poly_model_merge:
            p_w_path = ['./model/cross_validation/pl_SGDlr0.02_ers_b32_0.769/2_0.772/best_model.pth']
            poly_model_weight = torch.load(weights_poly_curve, map_location='cpu')['model']
            for p in p_w_path:
                temp_weight = torch.load(p, map_location='cpu')['model']
                for weight_name in poly_model_weight:
                    assert weight_name in temp_weight.keys(), '模型不匹配'
                    poly_model_weight[weight_name] = (poly_model_weight[weight_name] + temp_weight[weight_name]) / 2
            model_poly_curve.load_state_dict(poly_model_weight)
        else:
            model_poly_curve.load_state_dict(torch.load(weights_poly_curve, map_location='cpu')['model'])
        model_poly_curve.to(device)
        model_poly_curve.eval()

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
            img_name = json_dir.split(run_env)[-1].split('_jpg')[0].split('_JPG')[0]

            assert img_name in spacing_cm['name'], img_name + ' not in spacing_cm'
            sp_cm = spacing_cm['spa'][spacing_cm['name'].index(img_name)] * 10  # * 10为mm，否则为cm

            original_img, ROI_img, to_pre_img, ROI_target, box, resize_ratio = test_data[index]
            towards_right = test_data.towards_right
            # expand batch dimension
            to_pre_img = torch.unsqueeze(to_pre_img, dim=0)

            # heatmap model 预测六个点
            output = model(to_pre_img.to(device))
            prediction = output.squeeze().to('cpu')
            # 去除pad的数据
            raw_w, raw_h = ROI_img.shape[-1], ROI_img.shape[1]
            ROI_target['mask'] = ROI_target['mask'][:, :raw_h, :raw_w]
            prediction = prediction[:, :raw_h, :raw_w]

            # 计算预测数据的landmark 的 mse误差
            mse['name'].append(img_name)
            for i, data in enumerate(prediction[prediction.shape[0] - 6:]):
                y, x = np.where(data == data.max())
                point = ROI_target['landmark'][i + 8]  # label=i+8
                error = round(math.sqrt(math.pow(x[0] - point[0], 2) + math.pow(y[0] - point[1], 2)), 3)
                mse[name_index[i + 8]].append(error*sp_cm)

            # poly model 预测poly_curve
            if model2:
                output2 = model_poly_curve(to_pre_img.to(device))
                prediction2 = output2['out'].to('cpu')
                # 去除Pad的数据
                prediction2 = prediction2[:, :, :raw_h, :raw_w]

                # 保存预测的热力图结果
                if not os.path.exists(os.path.join(save_root, 'pre_heatmap')):
                    os.makedirs(os.path.join(save_root, 'pre_heatmap'))
                for sssj, ssss in enumerate(prediction):
                    plt.imsave(os.path.join(save_root, 'pre_heatmap', img_name + '_' + str(sssj) + '.png'), ssss)

                # 计算预测的dice
                dice = multiclass_dice_coeff(torch.nn.functional.softmax(prediction2, dim=1),
                                             ROI_target['mask'].unsqueeze(0))
                for i, (k, v) in enumerate(dices.items()):
                    if k == 'name':
                        dices['name'].append(img_name)
                    else:
                        dices[k].append(float(dice[i-1]))
                # print(f'dice:{dice:.3f}')

                # 生成预测数据的统一格式的target{'landmark':landmark,'mask':mask}
                prediction = torch.cat((prediction2.squeeze(), prediction), dim=0)
                pre_ROI_target, not_exist_landmark = create_predict_target(ROI_img, prediction, json_dir,
                                                                           towards_right=towards_right, deal_pre=True)
                show_img(ROI_target['show_roi_img'], pre_ROI_target, ROI_target, title=img_name,
                         save_path=os.path.join(save_root, 'pre_results'))

                # 将ROI target 转换为原图大小
                target = create_origin_target(ROI_target, box, original_img.size)
                pre_target = create_origin_target(pre_ROI_target, box, original_img.size)

                if len(not_exist_landmark) > 0:
                    print(not_exist_landmark)
                    show_img(original_img, pre_target)
                # 分指标展示'IFA', 'MNM', 'FMA', 'PL', 'MML'
                # for metric in ['IFA', 'MNM', 'FMA', 'PL', 'MML']:
                #     show_one_metric(original_img, target, pre_target, metric, not_exist_landmark, show_img=True)
                # 计算颜面的各个指标
                pre_data, pre_img = calculate_metrics(original_img, pre_target, not_exist_landmark, is_gt=False,
                                                      resize_ratio=resize_ratio,  show_img=False, compute_MML=True,
                                                      spa=sp_cm, name=img_name, save_path=save_root + '/PRE')
                gt_data, gt_img = calculate_metrics(original_img, target, not_exist_landmark=[], show_img=False,
                                                    resize_ratio=resize_ratio, compute_MML=True, spa=sp_cm, 
                                                    name=img_name, save_path=save_root + '/GT')
                if img_name in ['0110062_Yue_Guo_20170822092632751',
                                '0115810_Cheng_Zhang_20200909143417688', '0116840_Yongping_Dai_20210203103117950']:
                    for i in ['IFA', 'MNM', 'FMA', 'PL', 'FS']:
                        show_one_metric(original_img, target, pre_target, i, not_exist_landmark, spa=sp_cm, box=box,
                                        resize_ratio=resize_ratio, save_path=f'{save_root}/{img_name}')
                # save results and predicted images
                result_pre['name'].append(img_name)
                result_gt['name'].append(img_name)
                for key in ['IFA', 'MNM', 'FMA', 'FPL', 'PL', 'MML', 'FS']:
                    errorkey = ''
                    result_pre[key].append(pre_data[key])
                    result_gt[key].append(gt_data[key])
                    if pre_data[key] != 'not' and gt_data[key] != 'not':
                        error = abs(round(gt_data[key] - pre_data[key], 3))
                    if key == 'IFA' and (error > 13.38 or error < -11.75):
                        errorkey = 'IFA'
                    if key == 'MNM' and (error > 4 or error < -3.45):
                        errorkey = 'MNM'
                    if key == 'FMA' and (error > 12.485 or error < -10.75):
                        errorkey = 'FMA'
                    if key == 'PL' and (error > 0.793 or error < -0.7):
                        errorkey = 'PL'
                    if key == 'FS' and (error > 1.985 or error < -2.0):
                        errorkey = 'FS'
                    if len(errorkey) > 0:
                        save_error_path = save_root + '/error'
                        save_error_path_ = os.path.join(save_error_path, errorkey)
                        if not os.path.exists(os.path.join(save_error_path, errorkey)):
                            os.makedirs(save_error_path_)
                        pre_img.save(os.path.join(save_error_path_, img_name + '_pre_' + str(float(error)) + '.png'))
                        gt_img.save(os.path.join(save_error_path_, img_name + '_gt_' + str(float(error)) + '.png'))

    # 评估 mse误差   var100_mse_Rightcrop最佳
    for i in mse:
        if i == 'name':
            continue
        dd = np.asarray(mse[i])
        print(i, dd.mean(), dd.std())
        for j in [0.2, 0.3, 0.4, 0.5]:
            print(j, len(np.where(dd < j)[0])/500)
    df = pd.DataFrame(mse)
    df.to_excel(save_root + '/mse_mm.xlsx')

    # 评估颜面误差
    if model2:
        # dice 误差
        mean_dice = [np.asarray(j).mean() for i, j in dices.items() if i != 'name']
        print(mean_dice)
        df = pd.DataFrame(dices)
        df.to_excel(save_root + '/dices.xlsx')
        
        # output the mean and std datas of each metrics
        for i in ['IFA', 'MNM', 'FMA', 'PL', 'FS']:
            pre = result_pre[i]
            gt = result_gt[i]
            print(i)
            # python 数组可以用count 不可以用 Ture or False作为索引， numpy 数组可以用True or False作为索引，不能用count
            print('没有预测：', pre.count(-1))
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
            print('error:', np.mean(error))
            print('error标准差:', np.std(error))
        for i in ['FPL', 'MML']:
            print(i)
            print('not  gt:', result_gt[i].count('not'), '    pre: ', result_pre[i].count('not'))
            print('阳性 1  gt:', result_gt[i].count(1), '    pre: ', result_pre[i].count(1))
            print('阴性 -1  gt:', result_gt[i].count(-1), '    pre: ', result_pre[i].count(-1))
            print('0  gt:', result_gt[i].count(0), '    pre: ', result_pre[i].count(0))
        
        # save 
        df_gt = pd.DataFrame(result_gt)
        df_gt.to_excel(save_root + '/result_gt.xlsx', index=True)
        df_pre = pd.DataFrame(result_pre)
        df_pre.to_excel(save_root + '/result_pre.xlsx', index=True)

        # 计算icc
        for metric in result_pre:
            if metric in ['IFA', 'MNM', 'FMA', 'FS', 'PL']:
                icc_1, icc_k = analyze.metric_icc(result_pre, result_gt, metric)
                print('{}: icc_1: {:.3f}   icc_k: {:.3f}'.format(metric, icc_1, icc_k))
                analyze.metric_BA(result_pre, result_gt, metric, save_path=save_root + '/BA/')


if __name__ == '__main__':
    main()
