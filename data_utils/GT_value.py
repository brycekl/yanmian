import os
import time

import matplotlib.pyplot as plt
from PIL import Image

from eva_utils.my_eval import *


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def show_img(img, target, title=''):
    img = np.array(img)
    mask = target['mask']
    landmark = target['landmark']
    # for j in range(1, 5):
    #     mask_ = np.where(mask == j)
    #     img[..., 0][mask_] = j * 15
    #     img[..., 1][mask_] = j * 30 + 50
    #     img[..., 2][mask_] = j * 15
    # for i in landmark.values():
    #     cv2.circle(img, i, 2, (0, 255, 0), -1)
    mask_ = np.where(mask == 1)
    img[..., 0][mask_] = 200
    img[..., 1][mask_] = 100
    img[..., 2][mask_] = 200
    plt.title(title)
    plt.imshow(img)
    plt.show()


'''
测量所有GT的指标值
'''


def main():  # exclude background
    # txt_path = 'all_has_crl.txt'
    save_path = '../result/test'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    txt_path = 'all.txt'
    with open(txt_path) as read:
        json_list = [os.path.join('../data/jsons', line.strip())
                     for line in read.readlines() if len(line.strip()) > 0]

    # get CRL data
    import pandas as pd
    crl_datas = pd.read_excel('./data_information/CRL.xlsx')
    crl_datas = crl_datas[['seq', 'CRL']]
    crl_datas = crl_datas.to_dict('list')
    spa_excel = pd.read_excel('./data_information/spacing_values.xlsx').to_dict('list')
    spa_excel['name'] = [i.split('.')[0] for i in spa_excel['name']]
    # get devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    result_gt = {'name': [], 'IFA': [], 'MNM': [], 'FMA': [], 'FPL': [], 'PL': [], 'MML': [], 'FS': []}

    for index, json_dir in enumerate(json_list):
        if index % 10 == 0:
            print('\rnumber : ' + str(index), end='')
        json_str = open(json_dir, 'r', encoding='utf8')
        json_data = json.load(json_str)
        json_str.close()

        img_name = json_data['FileInfo']['Name']
        img_path = os.path.join('../data/images', img_name)

        # crl = crl_datas['CRL'][crl_datas['seq'].index(img_name.split('.')[0])]
        # if isinstance(crl, str):
        #     crl = int(crl.split('/')[0])
        spa = spa_excel['spa'][spa_excel['name'].index(img_name.split('.')[0])] * 10
        origin_image = Image.open(img_path)

        roi_target = {}
        # get curve, landmark data
        temp_curve = json_data['Models']['PolygonModel2']  # list   [index]['Points']
        curve = []
        # 去除标curve时多标的点
        for temp in temp_curve:
            if len(temp['Points']) > 2:
                curve.append(temp)
        landmark = json_data['Models']['LandMarkListModel']['Points'][0]['LabelList']
        # 将landmark转换为int，同时去除第三个维度
        landmark = {i['Label']: np.array(i['Position'], dtype=np.int32)[:2] for i in landmark}
        poly_points = json_data['Polys'][0]['Shapes']
        # get polygon mask
        mask_path = os.path.join('../data/masks', json_dir.split('\\')[-1].split('.')[0] + '_mask_255.jpg')
        # !!!!!!!!! np会改变Image的通道排列顺序，Image，为cwh，np为hwc，一般为chw（cv等等） cv:hwc
        mask_img = Image.open(mask_path)  # Image： c, w, h
        mask_array = np.array(mask_img)  # np 会将shape变为 h, w, c

        curve = {i['Label']: [[j[0], j[1]] for j in i['Points']] for i in curve}
        # 生成poly_curve 图-->cv2无法使用tensor
        poly_curve = np.zeros_like(mask_array)
        for i in range(197, 210):  # 上颌骨区域
            poly_curve[mask_array == i] = 1
        for i in range(250, 256):  # 下颌骨区域
            poly_curve[mask_array == i] = 2
        for label, points in curve.items():
            points_array = np.array(points, dtype=np.int32)
            for j in range(len(points_array) - 1):
                cv2.line(poly_curve, points_array[j], points_array[j + 1], color=label - 3, thickness=2)
        # as_tensor共享内存，tensor()则复制一份
        poly_curve = torch.as_tensor(poly_curve)
        roi_target['landmark'] = landmark
        roi_target['mask'] = poly_curve

        # show_img(origin_image, roi_target)
        # 分指标展示'IFA', 'MNM', 'FMA', 'PL', 'MML'
        # for metric in ['IFA', 'MNM', 'FMA', 'PL', 'MML']:
        #     show_one_metric(original_img, target, pre_target, metric, not_exist_landmark, show_img=True)
        # 计算颜面的各个指标
        gt_data, iiimg = calculate_metrics(origin_image, roi_target, not_exist_landmark=[], show_img=False, spa=spa,
                                           compute_MML=True, name=img_name.split('.')[0],  save_path=save_path)

        result_gt['name'].append(img_name.split('.')[0])
        for key in ['IFA', 'MNM', 'FMA', 'FPL', 'PL', 'MML', 'FS']:
            result_gt[key].append(gt_data[key])

    print('共{}张有效预测图片'.format(len(result_gt['IFA'])))
    print('         max       min       mean      std')
    for cat in result_gt:
        data = result_gt[cat]
        if cat not in ['FPL', 'MML', 'name']:
            print('{:<6}   {:<6}    {:<6}    {:<6}    {:<6}'.format(cat, round(max(data), 2),
                                                                    round(min(data), 2) if min(data) > 0 else round(
                                                                        min(data), 1),
                                                                    round(float(np.mean(data)), 2),
                                                                    round(float(np.std(data)), 2)))
    for cat in result_gt:
        if cat in ['FPL', 'MML', 'name']:
            print(cat, '    阳性1:', result_gt[cat].count(1), '    0:', result_gt[cat].count(0), '    阴性-1:',
                  result_gt[cat].count(-1))

    save_result = True
    if save_result is True:
        import pandas as pd
        df = pd.DataFrame(result_gt)
        df.to_excel(save_path + '/result_gt_2370.xlsx', sheet_name='Sheet1', index=False)

    # 评估颜面误差


if __name__ == '__main__':
    main()
