"""
对预测结果进行显著性分析
使用t检验或者Mann Whitney U检验
"""
import cv2
import pandas as pd
from scipy import stats
from scipy.stats import kstest, shapiro, normaltest, anderson
import numpy as np

# 数据信息
len_data = {'gt': 2372, '18': 20, '21': 18, 'clp': 20}   # 每个excel数据的个数
metrics = ['IFA', 'MNM', 'FMA', 'PL', 'FS']   # 需要分析的指标
final_result = {d_type: {metric: {'norm_p': 0, 't_type': '', 'value': 0, 'p': 0, 'conclusion': True} if d_type != 'gt'
                         else {'norm_p': 0} for metric in metrics} for d_type in ['gt', '18', '21', 'clp']}

# 获取数据
gt = pd.read_excel('./data_information/result_gt_2370_mm.xlsx', sheet_name='gt')
pre = pd.read_excel('./result_230103_mm/result_pre_.xlsx', sheet_name=None)
datas = {d_type: pre[d_type].to_dict('list') for d_type in ['18', '21', 'clp']}
datas['gt'] = gt.to_dict('list')

# lunwen 2
# datas = pd.read_excel('./datas/data_information/lunwen2_result.xlsx', sheet_name=None)
# datas = {d_type: datas[d_type].to_dict('list') for d_type in ['gt', '18', '21', 'clp']}
for d_type in len_data:
    for metric in metrics:
        datas[d_type][metric] = datas[d_type][metric][:len_data[d_type]]

# 对每个指标进行正态性检验
for d_type in final_result:
    for metric in final_result[d_type]:
        data = np.array(datas[d_type][metric])
        # test_result = kstest(data, 'norm', alternative='less')
        test_result = shapiro(data)
        # test_result = normaltest(data)
        final_result[d_type][metric]['norm_p'] = round(test_result[1], 3)

# 对各指标中，符合正太分布的数据进行独立样本t检验，否则进行Mann-Whitney检验
for d_type in ['18', '21', 'clp']:
    for metric in final_result[d_type]:
        gt_data = datas['gt'][metric]
        metric_data = datas[d_type][metric]
        if final_result['gt'][metric]['norm_p'] > 0.05 and final_result[d_type][metric]['norm_p'] > 0.05:
            # t test
            # 方差齐性检验
            _, levene_p = stats.levene(gt_data, metric_data)
            v, p = stats.ttest_ind(gt_data, metric_data, equal_var=True if levene_p > 0.05 else False)
            final_result[d_type][metric]['t_type'] = 't_test'
        else:
            # Mann-Whitney test
            v, p = stats.mannwhitneyu(gt_data, metric_data, alternative='two-sided')
            final_result[d_type][metric]['t_type'] = 'M-W U_test'
        final_result[d_type][metric]['value'] = round(v, 3)
        final_result[d_type][metric]['p'] = round(p, 3)
        final_result[d_type][metric]['conclusion'] = True if p < 0.05 else False

# 打印最后结果
for d_type in ['18', '21', 'clp']:
    print(d_type)
    for metric in final_result[d_type]:
        t_re = final_result[d_type][metric]
        print('{}   test type: {}    test value: {}    P: {}    conclusion: {}'.format
              (metric, t_re['t_type'], t_re['value'], t_re['p'], t_re['conclusion']))


# 写入excel
# with pd.ExcelWriter('./result_1208/test_result.xlsx') as writer:
#     for d_type in final_result:
#         df = pd.DataFrame(final_result[d_type])
#         df.to_excel(writer, sheet_name=d_type)
# 1208 1221 1304 1623
# 分析预测结果和医生的一致性
from eva_utils import analyze
doctor = pd.read_excel('./result_1213_mm/result_doctor_2.xlsx', sheet_name=None)
doctors = {d_type: doctor[d_type].to_dict('list') for d_type in ['18', '21', 'clp']}
for d_type in ['18', '21', 'clp']:
    for metric in metrics:
        doctors[d_type][metric] = doctors[d_type][metric][:len_data[d_type]]
for d_type in ['18', '21', 'clp']:
    print(d_type)
    for metric in metrics:
        if metric not in ['name', 'FPL', 'MML']:
            icc_1, icc_k = analyze.metric_icc(pre[d_type], doctors[d_type], metric)
            print('{}: icc_1: {:.3f}   icc_k: {:.3f}'.format(metric, icc_1, icc_k))
