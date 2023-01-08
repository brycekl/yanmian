import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

result_gt = pd.read_excel('./result2/result_gt.xlsx')
result_pre = pd.read_excel('./result2/result.xlsx', None)

result_gt = result_gt.to_dict('list')
result_pre = {i: result_pre[i].to_dict('list') for i in result_pre}
index = {'21': '21三体', '18': '18三体', 'NSCLP': '唇腭裂'}
# for label in result_pre:
#     print(label, '共{}张有效预测图片'.format(len(result_pre[label]['IFA'])))
#     print('        max     min     mean    std')
#     for metric in result_pre[label]:
#         if metric != 'name':
#             data = result_pre[label][metric]
#             print('{:<6}  {:<5}   {:<5}   {:<5}   {:<5}'.format(metric, max(data), min(data),
#                                                                 round(float(np.mean(data)), 2),
#                                                                 round(float(np.std(data)), 2)))

plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False

for metric in ['IFA', 'MNM', 'FMA', 'PL', 'FS']:
    crl = result_gt['CRL']
    data = result_gt[metric]
    # model = LinearRegression()
    # model.fit(crl, result_gt[i])
    plt.scatter(crl, data, marker='o', color='gray')
    plt.scatter(result_pre['唇腭裂']['CRL'], result_pre['唇腭裂'][metric], color='darkorchid', marker='D')
    plt.scatter(result_pre['18三体']['CRL'], result_pre['18三体'][metric], color='lime', marker='^')
    plt.scatter(result_pre['21三体']['CRL'], result_pre['21三体'][metric], color='red', marker='s')
    plt.legend(['正常', '唇腭裂', '18三体', '21三体'])
    plt.xlabel('CRL(mm)')
    if metric in ['PL', 'FS']:
        plt.ylabel(metric + '(pixel)')
    else:
        plt.ylabel(metric + '(°)')
    # 绘制
    poly_ = np.polyfit(crl, data, 1)
    curve = np.poly1d(poly_)
    plt.plot(crl, curve(crl), color='black')

    # 绘制 95% 预测区间
    y_curve = curve(crl)
    y_shift = data - y_curve
    arc = math.atan(poly_[0])
    distance = sorted(y_shift)
    less, more = 0, 0
    for i in distance:
        if i < 0:
            less += 1
        elif i > 0:
            more += 1
    # 下面的线
    print(int(less * 0.05), int(more * 0.05))
    yuan = poly_[1]
    poly_[1] = yuan + distance[int(less * 0.05)]
    curve_ = np.poly1d(poly_)
    plt.plot(crl, curve_(crl), color='black', linestyle='dotted')
    # 上面的线
    poly_[1] = yuan + distance[len(distance) - int(more * 0.05)]
    curve_ = np.poly1d(poly_)
    plt.plot(crl, curve_(crl), color='black', linestyle='dotted')
    plt.show()
