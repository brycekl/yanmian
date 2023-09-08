import os

import matplotlib.pyplot as plt

import numpy as np


def icc_calculate(Y, icc_type):
    assert icc_type in ['icc(1)', 'icc(2)', 'icc(3)'], "icc_type must in ['icc(1)', 'icc(2)', 'icc(3)']"
    [n, k] = Y.shape

    # 采用一致性分析（非绝对一致性）
    # 自由度
    dfall = n * k - 1  # 所有自由度
    dfe = (n - 1) * (k - 1)  # 剩余自由度
    dfc = k - 1  # 列自由度
    dfr = n - 1  # 行自由度

    # 所有的误差
    mean_Y = np.mean(Y)
    SST = ((Y - mean_Y) ** 2).sum()

    x = np.kron(np.eye(k), np.ones((n, 1)))  # sessions
    x0 = np.tile(np.eye(n), (k, 1))  # subjects
    X = np.hstack([x, x0])

    # 误差均方
    predicted_Y = np.dot(
        np.dot(np.dot(X, np.linalg.pinv(np.dot(X.T, X))), X.T), Y.flatten("F")
    )
    residuals = Y.flatten("F") - predicted_Y
    SSE = (residuals ** 2).sum()

    MSE = SSE / dfe

    # 列均方
    SSC = ((np.mean(Y, 0) - mean_Y) ** 2).sum() * n
    MSC = SSC / dfc

    # 行均方
    SSR = ((np.mean(Y, 1) - mean_Y) ** 2).sum() * k
    MSR = SSR / dfr

    if icc_type == "icc(1)":
        SSW = SST - SSR  # 剩余均方
        MSW = SSW / (dfall - dfr)

        ICC1 = (MSR - MSW) / (MSR + (k - 1) * MSW)
        ICC2 = (MSR - MSW) / MSR

    elif icc_type == "icc(2)":

        ICC1 = (MSR - MSE) / (MSR + (k - 1) * MSE + k * (MSC - MSE) / n)
        ICC2 = (MSR - MSE) / (MSR + (MSC - MSE) / n)

    elif icc_type == "icc(3)":

        ICC1 = (MSR - MSE) / (MSR + (k - 1) * MSE)
        ICC2 = (MSR - MSE) / MSR

    return ICC1, ICC2


def metric_icc(result_pre, result_gt, metric, icc_type="icc(3)"):
    assert metric in ['IFA', 'MNM', 'FMA', 'PL', 'FPL', 'MML', 'FS']
    data = [[], []]
    for p, g in zip(result_pre[metric], result_gt[metric]):
        data[0].append(p)
        data[1].append(g)
    data = np.array(data)
    data = data.T
    icc_1, icc_k = icc_calculate(data, icc_type)
    return icc_1, icc_k


def metric_BA(result_pre, result_gt, metric, save_path=None):
    # plt.rc('font', family='Times New Roman')
    font_title = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20}
    font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 16}
    font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 10}
    assert metric in ['IFA', 'MNM', 'FMA', 'PL', 'FPL', 'MML', 'FS']
    show_metrics = {'IFA': 'IFA', 'MNM': 'MNM angle', 'FMA': 'FMA', 'FS': 'FS distance', 'PL': 'PL distance'}
    pre_data = np.array(result_pre[metric])
    gt_data = np.array(result_gt[metric])
    error = gt_data - pre_data
    error_x = (gt_data + pre_data) / 2
    error_mean = np.mean(error)
    error_std = np.std(error)
    p_mean, p_std = pre_data.mean(), pre_data.std()
    g_mean, g_std = gt_data.mean(), gt_data.std()
    print(metric)
    print(p_mean, p_std, p_mean-1.96*p_std, p_mean+1.96*p_std)
    print(g_mean, g_std, g_mean-1.96*g_std, g_mean+1.96*g_std)
    print(error_mean, error_std, error_mean-1.96*error_std, error_mean+1.96*error_std)

    # le为各指标偏移值，（上下偏移和左右偏移）
    # mean+SD 左右，mean-SD 左右，上下，1.96SD左右，-1.96SD左右，上下
    le = {'IFA': [7.8, 7.3, 1, 1.1, 1.7, 2.1], 'MNM': [1.9, 1.8, 0.25, -0.13, 0.25, 0.6], 'FMA': [6.65, 6.3, 1.1, 1.1, 1.5, 2],
          'PL': [0.9, 0.84, 0.05, 0.05, 0.02, 0.11], 'FS': [1.35, 1.27, 0.2, 0.1, 0.15, 0.38]}
    plt.scatter(error_x, error, marker='.', linewidths=2)
    plt.axhline(error_mean, linestyle='-', color='b')
    plt.axhline(error_mean + 1.96 * error_std, linestyle='--', color='r')
    plt.axhline(error_mean - 1.96 * error_std, linestyle='--', color='r')
    plt.text(error_x.max()-le[metric][0], error_mean + 1.96 * error_std + le[metric][2], str('mean+1.96 SD'),
             fontdict=font2)
    plt.text(error_x.max()-le[metric][3], error_mean + 1.96 * error_std - le[metric][5],
             str(round(float(error_mean + 1.96 * error_std), 2)), fontdict=font2)
    plt.text(error_x.max()-le[metric][1], error_mean - 1.96 * error_std + le[metric][2], str('mean-1.96 SD'),
             fontdict=font2)
    plt.text(error_x.max()-le[metric][4], error_mean - 1.96 * error_std - le[metric][5],
             str(round(float(error_mean - 1.96 * error_std), 2)), fontdict=font2)

    num_95 = 0
    for e in error:
        if e > error_mean + 1.96 * error_std or e < error_mean - 1.96 * error_std:
            num_95 += 1
    # print(metric, 'mean:{},  std:{},  有{}个值位于95置信区间外'.format(error_mean, error_std, num_95))
    plt.title(metric, fontdict=font_title)
    plt.xlabel("Mean of AI and Manual", fontdict=font1)
    plt.ylabel("(Manual - AI) / Mean %", fontdict=font1)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.yticks(fontproperties='Times New Roman', size=14)

    # plt.title('AI group and manual measurement group \n consistency of {}'.format(show_metrics[metric]),
    #           fontdict=font1)

    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, metric) + '.png')
        plt.close()
    else:
        plt.show()

    # plt.figure(figsize=(7, 6), dpi=100)
    # ax = plt.gca()
    # plt.title('78')
    # sm.graphics.mean_diff_plot(np.array(a1), np.array(a2), sd_limit=1.96, ax=ax,
    #                            scatter_kwds=dict(color='deepskyblue'), mean_line_kwds=dict(color='red'),
    #                            limit_lines_kwds=dict(color='black', linewidth=1.5))


def compute_pck():
    pass