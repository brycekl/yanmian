import src
import torch


def create_lr_scheduler(optimizer,
                        epochs: int,  # 总的epoch 数
                        lr_name='multiStepLR',
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        **lr_config):
    """
    此处只定义普通的不带warmup的lr schedule，warmup单独在训练时由第一个epoch实现
    """
    if warmup is False:
        warmup_epochs = 0

    if lr_name == 'MultiStepLR':
        assert 'milestones' in lr_config and 'gamma' in lr_config
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_config['milestones'], lr_config['gamma'])
    elif lr_name == 'ConstantLR':
        assert 'T_max' in lr_config
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, lr_config['T_max'])
    elif lr_name == 'my_lr':
        # 每个epoch lr 都会减小
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda epoch: (1-(epoch-warmup_epochs)/(epochs-warmup_epochs)) ** 0.9)


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        """根据step数返回一个学习率倍率因子"""
        if x >= warmup_iters:  # 当迭代数大于给定的warmup_iters时，倍率因子为1
            return 1
        alpha = float(x) / warmup_iters
        # 迭代过程中倍率因子从warmup_factor -> 1
        return warmup_factor * (1 - alpha) + alpha
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def create_model(num_classes_1, num_classes_2=0, in_channel=3, base_c=32, model_name='unet', **kwargs):
    if model_name.find('unet') != -1:
        model = src.UNet(in_channels=in_channel, num_classes=num_classes_1, num_classes_2=num_classes_2, base_c=base_c,
                         model_name=model_name)
    elif model_name == 'mobilev3unet':
        model = src.MobileV3Unet(num_classes=num_classes_1)
    elif model_name == 'vgg16unet':
        model = src.VGG16UNet(num_classes=num_classes_1)
    elif model_name.find('u2net') != -1:
        assert model_name in ['u2net_lite', 'u2net_full']
        model = src.u2net_list[model_name](num_classes_1)
    elif model_name == 'resnet34unet':
        model = src.Resnet34(3, num_classes_1)
    elif model_name.find('ViT') != -1:
        assert model_name in ['ViT-B_16', 'ViT-B_32', 'ViT-L_16', 'ViT-L_32', 'ViT-H_14', 'R50-ViT-B_16', 'R50-ViT-L_16']
        vit_config = src.CONFIGS[model_name]
        vit_config.n_classes = num_classes_1
        if model_name.find('R50') != -1:
            vit_config.patches.grid = (
                int(256 / 16), int(256 / 16))
        model = src.TransUNet(vit_config, img_size=kwargs['input_size'], num_classes=num_classes_1)

    # 下面是分类模型
    elif model_name.find('resnet') != -1:
        assert model_name in ['resnet_34', 'resnet_50', 'resnet_101']
        model = src.resnet[model_name](num_classes_1)
    elif model_name.find('densenet') != -1:
        assert model_name in ['densenet_121', 'densenet_161', 'densenet_169', 'densenet_201']
        model = src.densenet[model_name](num_classes=num_classes_1)
    elif model_name.find('efficientnetv2') != -1:
        assert model_name in ['efficientnetv2_s', 'efficientnetv2_m', 'efficientnetv2_l']
        model = src.efficientnetv2[model_name](num_classes=num_classes_1)
    elif model_name.find('efficientnet') != -1:
        assert model_name in ['efficientnet_b' + str(i) for i in range(8)]
        model = src.efficientnet[model_name](num_classes=num_classes_1)
    elif model_name.find('MobileNetV2') != -1:
        model = src.MobileNetV2(num_classes=num_classes_1)
    # model = HighResolutionNet(num_joints=num_classes, base_channel=base_c)
    print('create {} model successfully.'.format(model_name))
    return model


if __name__ == '__main__':
    s = 1