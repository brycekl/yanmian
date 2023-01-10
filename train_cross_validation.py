import datetime
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

import transforms as T
from src import VGG16UNet
from train_utils import train_one_epoch, evaluate, create_lr_scheduler, init_distributed_mode, save_on_master, mkdir
from yanMianDataset import YanMianDataset


class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, vflip_prob=0.5,
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
            T.RandomResize(min_size, max_size),
            T.RandomHorizontalFlip(0.5),
            T.RandomVerticalFlip(0.5),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            # T.RightCrop(2/3),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(train, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    base_size = 256
    crop_size = 480

    if train:
        return SegmentationPresetTrain(base_size, crop_size, mean=mean, std=std)
    else:
        return SegmentationPresetEval(mean=mean, std=std)


def create_model(num_classes):
    # model = UNet(in_channels=3, num_classes=num_classes, base_c=64)
    # model = MobileV3Unet(num_classes=num_classes)
    model = VGG16UNet(num_classes=num_classes)
    return model


def main(args):
    init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)
    # segmentation nun_classes + background
    num_classes = args.num_classes

    mean = (0.2281, 0.2281, 0.2281)
    std = (0.2313, 0.2313, 0.2313)

    # 用来保存coco_info的文件
    k = 5
    best_mses = []
    for ki in range(k):
        k_dir = args.output_dir + '/' + str(ki)
        if not os.path.exists(k_dir):
            os.mkdir(k_dir)
        name = args.output_dir.split('/')[-1] + str(ki)

        # 用来保存coco_info的文件
        # name = args.output_dir.split('/')[-1]
        results_file = args.output_dir + '/' + name + ".txt"

        train_dataset = YanMianDataset(args.data_path,
                                       data_type='train', json_path='./data/check_jsons',
                                       mask_path='./data/check_masks', txt_path='./data/cross_val.txt', ki=ki, k=k,
                                       transforms=get_transform(train=True, mean=mean, std=std), resize=[256, 256])

        val_dataset = YanMianDataset(args.data_path,
                                     data_type='val', json_path='./data/check_jsons',
                                     mask_path='./data/check_masks', txt_path='./data/cross_val.txt', ki=ki, k=k,
                                     transforms=get_transform(train=False, mean=mean, std=std), resize=[256, 256])

        print("Creating data loaders")
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            test_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        else:
            train_sampler = torch.utils.data.RandomSampler(train_dataset)
            test_sampler = torch.utils.data.SequentialSampler(val_dataset)

        train_data_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size,
            sampler=train_sampler, num_workers=args.workers,
            collate_fn=train_dataset.collate_fn, drop_last=False)

        val_data_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=1,
            sampler=test_sampler, num_workers=args.workers,
            collate_fn=train_dataset.collate_fn)

        print(len(train_dataset), len(val_dataset))
        print("Creating model")
        # create model num_classes equal background + foreground classes
        model = create_model(num_classes=num_classes)
        model.to(device)

        if args.sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        model_without_ddp = model
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            model_without_ddp = model.module

        params_to_optimize = [p for p in model_without_ddp.parameters() if p.requires_grad]

        # optimizer = torch.optim.SGD(
        #     params_to_optimize,
        #     lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        optimizer = torch.optim.Adam(params_to_optimize, weight_decay=args.weight_decay)  # lr = 2e-4
        # optimizer = torch.optim.NAdam(params_to_optimize, weight_decay=args.weight_decay)

        scaler = torch.cuda.amp.GradScaler() if args.amp else None

        # 创建学习率更新策略，这里是每个step更新一次(不是每个epoch)
        lr_scheduler = create_lr_scheduler(optimizer, len(train_data_loader), args.epochs, warmup=True)

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

        if args.test_only:
            confmat = evaluate(model, val_data_loader, device=device, num_classes=num_classes)
            val_info = str(confmat)
            print(val_info)
            return

        print("Start training")
        start_time = time.time()
        best_mse = 1000
        best_mse_weight = 1000
        train_losses = []
        val_losses = []
        mse_es = []
        best_epoch = {}
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                train_sampler.set_epoch(epoch)
            mean_loss, lr = train_one_epoch(model, optimizer, train_data_loader, device, epoch, num_classes,
                                            lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)

            val_loss, mse, dice = evaluate(model, val_data_loader, device=device, num_classes=num_classes)

            s_best = False
            save_best_weight = False
            m_mse = 0
            m_mse_weight = 0
            if args.save_best is True:
                for i in mse['mse_classes'].values():
                    m_mse += i
                m_mse = m_mse / len(mse['mse_classes'])
                if m_mse < best_mse:
                    best_mse = m_mse
                    s_best = True
                    if best_mse < 5:
                        best_epoch[epoch] = best_mse
                # 加权版
                weights = {8: 1.5, 9: 1, 10: 1, 11: 1.5, 12: 2, 13: 3}
                for index in mse['mse_classes']:
                    m_mse_weight += weights[index] * mse['mse_classes'][index]
                m_mse_weight = m_mse_weight / 10
                if m_mse_weight < best_mse_weight:
                    best_mse_weight = m_mse_weight
                    save_best_weight = True
                print('best mse : ', best_mse, 'best weight mse:', best_mse_weight)
                # else:
                #     continue

            # 只在主进程上进行写操作
            if args.rank in [-1, 0]:
                # write into txt
                with open(results_file, "a") as f:
                    # 记录每个epoch对应的train_loss、lr以及验证集各指标
                    train_info = f"[epoch: {epoch}]    lr: {lr:.6f}\n" \
                                 f"train_loss: {mean_loss:.4f}    val_loss: {val_loss:.4f}\n" \
                                 f"mse:{[round(mse['mse_classes'][i], 3) for i in range(8, 14)]}\n" \
                                 f"best_mse:{best_mse}    weight mse :{best_mse_weight}\n"
                    f.write(train_info + "\n\n")

                train_losses.append(round(float(mean_loss), 3))
                val_losses.append(round(float(val_loss), 3))
                mse_es.append(round(float(m_mse), 3))

                if args.output_dir:
                    # 只在主节点上执行保存权重操作
                    save_file = {'model': model_without_ddp.state_dict(),
                                 'optimizer': optimizer.state_dict(),
                                 'lr_scheduler': lr_scheduler.state_dict(),
                                 'args': args,
                                 'epoch': epoch}
                    if args.amp:
                        save_file["scaler"] = scaler.state_dict()

                    if save_best_weight is True:
                        save_on_master(save_file,
                                       os.path.join(k_dir, 'best_weight_model.pth'))
                        print('save best weight model')
                    if args.save_best is True and s_best is True:
                        save_on_master(save_file,
                                       os.path.join(k_dir, 'best_model.pth'))
                        print('save best model')
                        # save_on_master(save_file,
                        #                os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))

        if args.rank in [-1, 0]:
            # write into txt
            with open(results_file, "a") as f:
                # 记录每个epoch对应的train_loss、lr以及验证集各指标
                train_info = f"[best mse: {best_mse:.4f}]\n" \
                             f"[best mse weight: {best_mse_weight:.4f}]\n"
                train_info += f'epoch:mse    '
                for ep, va in best_epoch.items():
                    train_info += f"{ep}:{va}    "
                train_info += f'\n'
                f.write(train_info + "\n\n")

            train_losses = train_losses[2:]
            val_losses = val_losses[2:]
            mse_es = mse_es[2:]
            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print('Training time {}'.format(total_time_str))
            print('best_mse:', best_mse)
            print(best_epoch)
            plt.plot(train_losses, 'r', label='train_loss')
            plt.plot(val_losses, 'g', label='val_loss')
            plt.legend()
            plt.title(name)
            plt.savefig(args.output_dir + '/' + name + "_1.png")
            plt.close()
            plt.plot(mse_es, 'b', label='mse')
            plt.legend()
            plt.title(name)
            plt.savefig(args.output_dir + '/' + name + "_2.png")
            plt.close()
            best_mses.append(round(float(best_mse), 3))
            # 重命名
            new_name = k_dir + '_' + str(round(float(best_mse), 3))
            os.rename(k_dir, new_name)
    mean_mse = np.mean(best_mses)
    print(best_mses, mean_mse)
    new_name = args.output_dir + '_' + str(round(float(mean_mse), 3))
    os.rename(args.output_dir, new_name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    # 训练文件的根目录(DRIVE)
    parser.add_argument('--data-path', default='./', help='dataset')
    # 训练设备类型
    parser.add_argument('--device', default='cuda', help='device')
    # 检测目标类别数(不包含背景)
    parser.add_argument('--num-classes', default=6, type=int, help='num_classes')
    # 每块GPU上的batch_size
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    # 指定接着从哪个epoch数开始训练
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    # 训练的总epoch数
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    # 是否使用同步BN(在多个GPU之间同步)，默认不开启，开启后训练速度会变慢
    parser.add_argument('--sync_bn', type=bool, default=True, help='whether using SyncBatchNorm')
    # 数据加载以及预处理的线程数
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    # 训练学习率，这里默认设置成0.01(使用n块GPU建议乘以n)，如果效果不好可以尝试修改学习率
    parser.add_argument('--lr', default=0, type=float,
                        help='initial learning rate')
    # SGD的momentum参数
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    # SGD的weight_decay参数
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    # 只保存dice coefficient值最高的权重
    parser.add_argument('--save-best', default=True, type=bool, help='only save best weights')
    # 训练过程打印信息的频率
    parser.add_argument('--print-freq', default=1, type=int, help='print frequency')
    # 文件保存地址
    parser.add_argument('--output-dir', default='./model/cross_validation/hm_vu_hmprove_ad_m8_b8_',
                        help='path where to save')
    # 基于上次的训练结果接着训练
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    # 不训练，仅测试
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )

    # 分布式进程数
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    # 如果指定了保存文件地址，检查文件夹是否存在，若不存在，则创建
    if args.output_dir:
        mkdir(args.output_dir)

    main(args)
    # CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 train_cross_validation.py
