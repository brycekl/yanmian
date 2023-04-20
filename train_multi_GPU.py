import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import random

import transforms as T
from torch.utils.tensorboard import SummaryWriter
from train_utils import *
from yanMianDataset import YanMianDataset


class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, vflip_prob=0.5, hm_var=40,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.8 * base_size)
        max_size = int(1 * base_size)

        # 这些transforms都是自己写的  T.RandomResize(min_size, max_size)
        # 将图片左边和右边裁去1/6，下方裁去1/3
        # trans = [T.MyCrop(left_size=1/6,right_size=1/6, bottom_size=1/3)]
        # trans = [T.RightCrop(2/3)]
        trans = []
        # if hflip_prob > 0:
        #     trans.append(T.RandomHorizontalFlip(hflip_prob))
        trans.extend([
            T.GetROI(border_size=30),
            # T.RandomResize(min_size, max_size, resize_ratio=1, shrink_ratio=1),
            T.AffineTransform(rotation=(-10, 10), input_size=args.input_size, resize_low_high=[0.8, 1],
                              heatmap_shrink_rate=args.heatmap_shrink_rate),
            T.RandomHorizontalFlip(0.5),
            T.RandomVerticalFlip(0.5),
            T.CreateGTmask(hm_var=hm_var),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            T.MyPad(size=256),
        ])

        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), hm_var=40):
        self.transforms = T.Compose([
            T.GetROI(border_size=30),
            T.AffineTransform(input_size=args.input_size, heatmap_shrink_rate=args.heatmap_shrink_rate),
            # T.RandomResize(256, 256, shrink_ratio=0),
            T.CreateGTmask(hm_var=hm_var),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            T.MyPad(size=256),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(train, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), hm_var=40):
    base_size = 256
    crop_size = 480

    if train:
        return SegmentationPresetTrain(base_size, crop_size, mean=mean, std=std, hm_var=hm_var)
    else:
        return SegmentationPresetEval(mean=mean, std=std, hm_var=hm_var)


def main(args):
    same_seeds(0)
    init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)
    # segmentation nun_classes + background
    num_classes1, num_classes2 = args.num_classes, args.num_classes2
    assert num_classes1 in [5, 6, 11] and num_classes2 in [0, 5]
    num_classes = num_classes1 + num_classes2

    mean = (0.2281, 0.2281, 0.2281)
    std = (0.2313, 0.2313, 0.2313)

    model_name = args.model_name
    output_dir = args.output_dir

    # 用来保存coco_info的文件
    # name = output_dir.split('/')[-1]
    results_file = output_dir + '/' + "log.txt"

    var = args.hm_var
    train_dataset = YanMianDataset(args.data_path, data_type='train', num_classes=num_classes,
                                   transforms=get_transform(train=True, mean=mean, std=std, hm_var=var))
    val_dataset = YanMianDataset(args.data_path, data_type='val', num_classes=num_classes,
                                 transforms=get_transform(train=False, mean=mean, std=std, hm_var=var))

    print("Creating data loaders")
    # 将数据打乱后划分到不同的gpu上
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        test_sampler = torch.utils.data.SequentialSampler(val_dataset)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.workers,
        collate_fn=train_dataset.collate_fn, drop_last=False, worker_init_fn=seed_worker)

    val_data_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, sampler=test_sampler, num_workers=args.workers, worker_init_fn=seed_worker,
        collate_fn=train_dataset.collate_fn)

    print(len(train_dataset), len(val_dataset))
    print("Creating model")
    model = create_model(num_classes_1=num_classes1, num_classes_2=num_classes2, in_channel=3, base_c=32,
                         model_name=model_name, input_size=args.input_size)
    model.to(device)

    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    params_to_optimize = [p for p in model_without_ddp.parameters() if p.requires_grad]

    # optimizer = torch.optim.SGD(params_to_optimize, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(params_to_optimize, lr=args.lr, weight_decay=args.weight_decay)  # lr = 2e-4
    # optimizer = torch.optim.NAdam(params_to_optimize, weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # 创建学习率更新策略，这里是每个step更新一次(不是每个epoch)
    lr_scheduler = create_lr_scheduler(optimizer, args.epochs, lr_name=args.scheduler, milestones=args.lr_milestones,
                                       gamma=args.lr_gamma)

    # 如果传入resume参数，即上次训练的权重地址，则接着上次的参数训练
    if args.resume:
        # If map_location is missing, torch.load will first load the module to CPU
        # and then copy each parameter to where it was saved,
        # which would result in all processes on the same machine using the same set of devices.
        checkpoint = torch.load(args.resume, map_location='cpu')  # 读取之前保存的权重文件(包括优化器以及学习率策略)
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    # 记录训练/测试的损失函数变化，
    # 记录验证获得最优的各点的mse，dice，以及mse变化、dice变化、取得最优指标时的epoch
    losses = {'train_losses': {'mse_loss': [], 'dice_loss': []}, 'val_losses': {'mse_loss': [], 'dice_loss': []}}
    metrics = {'best_mse': {8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 'm_mse': 1000}, 'best_mse_weight': 1000,
               'best_dice': 0, 'dice': [], 'mse': [], 'best_epoch_mse': {}, 'best_epoch_dice': {}}
    save_model = {'save_mse': False, 'save_mse_weight': False, 'save_dice': False}
    dice_weight = 5 if num_classes == 11 else 1  # gei dice loss 进行加强的倍数
    # tensorboard writer
    tr_writer = SummaryWriter(log_dir=os.path.join(output_dir, 'train'))
    val_writer = SummaryWriter(log_dir=os.path.join(output_dir, 'val'))
    init_img = torch.zeros((1, 3, 256, 256), device=device)
    tr_writer.add_graph(model, init_img)

    print("Start training")
    start_time = time_synchronized()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        mean_loss, lr = train_one_epoch(model, optimizer, train_data_loader, device, epoch, num_classes,
                                        weight=dice_weight, print_freq=args.print_freq, scaler=scaler)
        lr_scheduler.step()
        val_loss, val_mse = evaluate(model, val_data_loader, device=device, num_classes=num_classes, weight=dice_weight)

        # 根据验证结果，求得平均指标，并判断是否需要保存模型
        if num_classes == 6 or num_classes == 11:
            val_mean_mse = np.average(list(val_mse['mse_classes'].values()))
            if val_mean_mse < metrics['best_mse']['m_mse']:
                metrics['best_mse']['m_mse'] = val_mean_mse
                save_model['save_mse'] = True
                if metrics['best_mse']['m_mse'] < 5:
                    metrics['best_epoch_mse'][epoch] = round(val_mean_mse, 3)
                for ind, c_mse in val_mse['mse_classes'].items():
                    metrics['best_mse'][ind] = round(c_mse, 3)
            # 加权版
            weights = {8: 1, 9: 1, 10: 1.2, 11: 1.3, 12: 1, 13: 1}
            m_mse_weight = 0
            for index in val_mse['mse_classes']:
                m_mse_weight += weights[index] * val_mse['mse_classes'][index]
            m_mse_weight = m_mse_weight / 6.5
            if m_mse_weight < metrics['best_mse_weight']:
                metrics['best_mse_weight'] = m_mse_weight
                save_model['save_mse_weight'] = True
            print(f'best_mse:{metrics["best_mse"]["m_mse"]:.3f}    '
                  f'best_weight_mse:{metrics["best_mse_weight"]:.3f}', end='  ')
        if num_classes == 11 or num_classes == 5:
            val_dice = float(1 - val_loss['dice_loss'])
            if val_dice > metrics['best_dice']:
                save_model['save_dice'] = True
                metrics['best_dice'] = val_dice
                if metrics['best_dice'] > 0.5:
                    metrics['best_epoch_dice'][epoch] = round(val_dice, 3)
            print(f'best dice : {metrics["best_dice"]:.3f}', end='')
        print('')

        # 只在主进程上进行写操作， 将结果写入txt
        if args.rank in [-1, 0]:
            with open(results_file, "a") as f:
                # 记录每个epoch对应的train_loss、lr以及验证集各指标
                train_info = f"[epoch: {epoch}]    lr: {lr:.6f}\n"
                tr_writer.add_scalar('learning rate', lr, epoch)
                if num_classes == 5 or num_classes == 11:
                    train_info += f"t_dice_loss: {mean_loss['dice_loss']:.4f}    " \
                                  f"v_dice_loss: {val_loss['dice_loss']:.4f}    "
                    losses['train_losses']['dice_loss'].append(round(float(mean_loss['dice_loss']), 3))
                    losses['val_losses']['dice_loss'].append(round(float(val_loss['dice_loss']), 3))
                    metrics['dice'].append(round(float(val_dice), 3))
                    tr_writer.add_scalar('dice_loss', mean_loss['dice_loss'], epoch)
                    val_writer.add_scalar('dice_loss', val_loss['dice_loss'], epoch)
                    val_writer.add_scalar('val_dice', val_dice, epoch)

                if num_classes == 6 or num_classes == 11:
                    train_info += f"t_mse_loss: {mean_loss['mse_loss']:.4f}    " \
                                  f"v_mse_loss:{val_loss['mse_loss']:.4f}\n" \
                                  f"mse:{[round(val_mse['mse_classes'][i], 3) for i in range(8, 14)]}\n" \
                                  f"best_mse:{metrics['best_mse']['m_mse']} weight mse :{metrics['best_mse_weight']}"
                    losses['train_losses']['mse_loss'].append(round(float(mean_loss['mse_loss']), 3))
                    losses['val_losses']['mse_loss'].append(round(float(val_loss['mse_loss']), 3))
                    metrics['mse'].append(round(float(val_mean_mse), 3))
                    tr_writer.add_scalar('mse_loss', mean_loss['mse_loss'], epoch)
                    val_writer.add_scalar('mse_loss', val_loss['mse_loss'], epoch)
                    val_writer.add_scalar('val_mse', val_mean_mse, epoch)
                f.write(train_info + "\n\n\n")

            # 保存模型
            if output_dir:
                # 只在主节点上执行保存权重操作
                save_file = {'model': model_without_ddp.state_dict(),
                             'optimizer': optimizer.state_dict(),
                             'lr_scheduler': lr_scheduler.state_dict(),
                             'args': args,
                             'epoch': epoch}
                if args.amp:
                    save_file["scaler"] = scaler.state_dict()

                if args.save_best is True and save_model['save_mse'] is True:
                    save_on_master(save_file, os.path.join(output_dir, 'model.pth'))
                    print('save best model')
                if save_model['save_mse_weight'] is True and False:
                    save_on_master(save_file, os.path.join(output_dir, 'weight_model.pth'))
                    print('save best weight model')
                if save_model['save_dice'] is True and num_classes != 11:
                    save_on_master(save_file, os.path.join(output_dir, 'model.pth'))
                    print('save best dice model')
                    # if best_mse < 4:
                    #     save_on_master(save_file,
                    #                    os.path.join(output_dir, 'model_{}.pth'.format(epoch)))

    # 训练结束，将最优结果写入txt
    if args.rank in [-1, 0]:
        total_time = time_synchronized() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
        train_info = f'Training time {total_time_str}\n'
        with open(results_file, "a") as f:
            if num_classes == 6 or num_classes == 11:
                train_info += f"best mse: {metrics['best_mse']['m_mse']:.4f}     " \
                              f"best mse weight: {metrics['best_mse_weight']:.4f}\n"  \
                              f"mse:{[metrics['best_mse'][i] for i in range(8, 14)]}\n"
                train_info += f'epoch:mse    '
                for ep, va in metrics['best_epoch_mse'].items():
                    train_info += f"{ep}:{va}    "
                train_info += f'\n'
            if num_classes == 5 or num_classes == 11:
                train_info += f"best dice: {metrics['best_dice']:.4f}\n"
                for ep, va in metrics['best_epoch_dice'].items():
                    train_info += f"{ep}:{va}    "
                train_info += f'\n'
            f.write(train_info)
        print(f'best_mse: {metrics["best_mse"]["m_mse"]:.3f}   best_weight_mse: {metrics["best_mse_weight"]:.3f}   '
              f'best_dice: {metrics["best_dice"]:.3f}')
        print(metrics['best_epoch_mse'], metrics['best_epoch_dice'])

        # 最后的作图 loss， metric图，以及文件夹重命名
        skip_epoch = 1  # 前面训练不稳定，作图跳过的epoch数
        assert skip_epoch >= 0 and skip_epoch <= args.epochs
        if num_classes == 6 or num_classes == 11:
            plt.plot(losses['train_losses']['mse_loss'][skip_epoch:], 'r', label='train_mse_loss')
            plt.plot(losses['val_losses']['mse_loss'][skip_epoch:], 'g', label='val_mse_loss')
        if num_classes == 5 or num_classes == 11:
            plt.plot(losses['train_losses']['dice_loss'][skip_epoch:], 'r', label='train_dice_loss')
            plt.plot(losses['val_losses']['dice_loss'][skip_epoch:], 'b', label='val_dice_loss')
        plt.legend()
        plt.savefig(output_dir + '/' + "loss.png")
        plt.close()
        if num_classes == 6 or num_classes == 11:
            plt.plot(metrics['mse'][skip_epoch:], 'g', label='mse')
        if num_classes == 5 or num_classes == 11:
            plt.plot(metrics['dice'][skip_epoch:], 'b', label='dice')
        plt.legend()
        plt.savefig(output_dir + '/' + "metric.png")
        plt.close()

        # 重命名
        new_name = output_dir + f'_var{var}_{metrics["best_mse"]["m_mse"]:.3f}' if num_classes == 6 else \
            (output_dir + f'_{metrics["best_dice"]:.3f}' if num_classes == 5 else
             output_dir + f'_var{var}_{metrics["best_mse"]["m_mse"]:.3f}_w{dice_weight}_{metrics["best_dice"]:.3f}')
        os.rename(output_dir, new_name)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)

    '''basic parameter'''
    parser.add_argument('--num_classes', default=11, type=int, help='number of classes')  # 11 / 6 / 4
    parser.add_argument('--num_classes2', default=0, type=int, help='number of classes')  # 0 / 5
    parser.add_argument('--output_dir', default='./models', help='path where to save')
    parser.add_argument('--model_name', default='unet', help='the model name')
    parser.add_argument('--heatmap_shrink_rate', default=1, type=int)  # hrnet最后没有复原为原图大小
    parser.add_argument('-b', '--batch_size', default=32, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('--random_seed', default=0, type=int, help='set random seed')
    parser.add_argument('--hm_var', default=40, type=int, help='heatmap var set')

    '''model setting'''
    # parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int, help='input channels')
    parser.add_argument('--input_size', default=[256, 256], nargs='+', type=int, help='input model size: [h, w]')
    parser.add_argument('--data_path', default='./datas', help='dataset')
    parser.add_argument('--loss', default='BCEDiceLoss',)  # todo 未设置

    '''optimizer'''
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'],)
    parser.add_argument('--lr', default=0.001, type=float,
                        help='initial learning rate, 0.001 is the default value for training '
                             'on 4 gpus and 32 images_per_gpu')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float, metavar='W',
                        help='weight decay (default: 1e-4)', dest='weight_decay')

    '''learning scheduler'''
    parser.add_argument('--scheduler', default='MultiStepLR',   # cv 用CosineAnnealingLR
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR', 'my_lr'])
    # 针对torch.optim.lr_scheduler.MultiStepLR的参数
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--epochs', default=150, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--lr_milestones', default=[100, 130], nargs='+', type=int,
                        help='decrease lr every step-size epochs')
    parser.add_argument('--lr_gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--early_stopping', default=-1, type=int, metavar='N', help='early stopping (default: -1)')

    '''other parameter'''
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--print_freq', default=10, type=int, help='print frequency')
    parser.add_argument('--save_best', default=True)
    # 基于上次的训练结果接着训练
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--test_only', action="store_true", help="test only")
    # 开启的进程数(注意不是线程)
    parser.add_argument('--world_size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument("--sync_bn", default=True, action="store_true", help="Use sync batch norm")
    # 是否使用混合精度训练(需要GPU支持混合精度)
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    import yaml

    args = parse_args()
    # 如果指定了保存文件地址，检查文件夹是否存在，若不存在，则创建
    args.output_dir = os.path.join(args.output_dir, args.model_name, args.model_name)
    if args.output_dir:
        mkdir(args.output_dir)

    # 如果没有分开的decoder标志 ’up'，则将num_classes2置0
    if args.model_name.find('up') == -1 and args.num_classes2 != 0:
        args.num_classes, args.num_classes2 = args.num_classes + args.num_classes2, 0

    # 写入文件
    with open(os.path.join(args.output_dir, 'config.yml'), 'w') as f:
        yaml.dump(args, f)

    main(args)
    # CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 train_multi_GPU.py
