import math

import numpy as np
import torch
from torch import nn

import train_utils.distributed_utils as utils
from .dice_coefficient_loss import dice_loss, build_target
from train_utils.init_model_utils import warmup_lr_scheduler
from train_utils.post_process import get_heatmap_maximum
from transforms import affine_points


def criterion(inputs, target, num_classes: int = 2, ignore_index: int = -100, weight=1):
    losses = {'mse_loss': 0, 'dice_loss': 0}

    # 忽略target中值为255的像素，255的像素是目标边缘或者padding填充
    # 交叉熵损失：在通道方向softmax后，根据x的值计算
    # loss = nn.functional.cross_entropy(x, target, ignore_index=ignore_index, weight=loss_weight)  # 函数式API

    if num_classes == 11 or num_classes == 5:
        # 针对每个类别，背景，前景都需要计算他的dice系数
        class_index = 0 if num_classes == 5 else 6
        # 根据gt构建每个类别的矩阵
        dice_target = build_target(target[:, class_index, :], 5, ignore_index)  # B * C* H * W
        # 计算两区域和两曲线的dice
        losses['dice_loss'] += (dice_loss(inputs[:, class_index:, :], dice_target, multiclass=True,
                                          ignore_index=ignore_index))
    if num_classes == 11 or num_classes == 6:
        if ignore_index > 0:
            roi_mask = torch.ne(target[:, :6, :], ignore_index)
            pre = inputs[:, :6, :][roi_mask]
            target_ = target[:, :6, :][roi_mask]
        losses['mse_loss'] += nn.functional.mse_loss(pre, target_) * weight
        # 总的损失为： 整幅图像的交叉熵损失和所有类别的dice损失之和
    return losses


def evaluate(model, data_loader, device, num_classes, weight=1):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    loss = {'dice_loss': 0, 'mse_loss': 0}
    mse = {i: [] for i in range(8, 14)}
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, mask = image.to(device), target['mask'].to(device)
            output = model(image)

            # 计算 loss 和 metric
            # 点定位计算mse loss 和 mse 的metric； 分割计算dice
            if num_classes == 11 or num_classes == 6:
                roi_mask = torch.ne(mask[:, :6, :], 255)
                pre = output[:, :6, :][roi_mask]
                target_ = mask[:, :6, :][roi_mask]
                loss['mse_loss'] += nn.functional.mse_loss(pre, target_) * weight
                # 计算mse # todo 此处使用仿射变换，没有pad，后续更改
                pre_keypoint = get_heatmap_maximum(output[:, :6, :].cpu().numpy())[0]
                pre_keypoint = affine_points(pre_keypoint[0], target['reverse_trans'][0])
                gt_keypoint = target['raw_keypoint'][0]
                for i, (p_, g_) in enumerate(zip(pre_keypoint, gt_keypoint)):
                    mse[i + 8].append(math.sqrt(math.pow(p_[0]-g_[0], 2) + math.pow(p_[1]-g_[1], 2)))
            if num_classes == 11 or num_classes == 5:
                class_index = 0 if num_classes == 5 else 6
                dice_target = build_target(mask[:, class_index, :], 5, 255)  # B * C* H * W
                loss['dice_loss'] += (dice_loss(output[:, class_index:, :], dice_target, multiclass=True, ignore_index=255))

    loss = {i: j / len(data_loader) for i, j in loss.items()}
    m_mse = []
    if num_classes == 11 or num_classes == 6:
        m_mse = {i: np.average(j) for i, j in mse.items()}
        for i in m_mse:
            print(f'{i} : {m_mse[i]:.3f}')
    return loss, {'mse_total': mse, 'mse_classes': m_mse}


def train_one_epoch(model, optimizer, data_loader, device, epoch, num_classes, warmup=True, print_freq=10,
                    scaler=None, weight=1):
    model.train()
    # MetricLogger 度量记录器 :为了统计各项数据，通过调用来使用或显示各项指标，通过具体项目自定义的函数
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0 and warmup is True:  # 当训练第一轮（epoch=0）时，启用warmup训练方式，可理解为热身训练
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor=1.0 / 1000)

    # 每次遍历一个iteration
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, mask = image.to(device), target['mask'].to(device)
        # BatchResizeC = BatchResize(480)
        # image, target = BatchResizeC(image, target)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            assert num_classes == output.shape[1]
            # 计算损失
            loss = criterion(output, mask, num_classes=num_classes, ignore_index=255, weight=weight)

            # # 使用dsntnn计算loss
            # # landmark 的target [B, C, 2(x, y)]
            # landmark = target['landmark']
            # dsnt_landmark = torch.as_tensor([[[l[i][0], l[i][1]] for i in range(8, 14)]for l in landmark])
            # img_size = image.shape[-2:]
            # dsnt_landmark = (dsnt_landmark * 2 + 1) / int(img_size[0]) - 1
            # dsnt_landmark = dsnt_landmark.to(device)
            # # 生成dsntnn的预测坐标
            # heatmaps = dsntnn.flat_softmax(output['out'])
            # coor = dsntnn.dsnt(heatmaps)
            # euc_losses = dsntnn.euclidean_losses(coor, dsnt_landmark)
            # reg_losses = dsntnn.js_reg_losses(heatmaps, dsnt_landmark, sigma_t=1.0)
            # loss = dsntnn.average_loss(euc_losses + reg_losses)

        back_loss = loss['mse_loss'] + loss['dice_loss']
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(back_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # 反向传播梯度
            back_loss.backward()
            optimizer.step()

        if lr_scheduler is not None:  # 第一轮使用warmup训练方式
            lr_scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        metric_logger.update(lr=lr)
        if num_classes == 6 or num_classes == 11:
            metric_logger.update(mse_loss=loss['mse_loss'].item())
        if num_classes == 5 or num_classes == 11:
            metric_logger.update(dice_loss=loss['dice_loss'].item())

    return_loss = {}
    if num_classes == 6 or num_classes == 11:
        return_loss['mse_loss'] = metric_logger.meters["mse_loss"].global_avg
    if num_classes == 5 or num_classes == 11:
        return_loss['dice_loss'] = metric_logger.meters["dice_loss"].global_avg
    return return_loss, lr


