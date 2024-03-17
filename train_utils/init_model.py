import os
import torch
import numpy as np
import random
from src import UNet, u2net, MobileV3Unet, VGG16UNet, resnet_unet, U_ConvNext, ResenetUnet


def same_seeds(seed):
    print(f'set same seed {seed}')
    torch.manual_seed(seed)  # 设置CPU随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 为特定GPU设置随机种子
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
    np.random.seed(seed)  # 固定Numpy产生的随机数
    random.seed(seed)  # 设置整个Python基础环境中的随机种子
    if seed == 0:
        torch.backends.cudnn.benchmark = False  # 不使用选择卷积算法的机制，使用固定的卷积算法（可能会降低性能）
        torch.backends.cudnn.deterministic = True   # 只限制benchmark的确定性
        # torch.use_deterministic_algorithms(True)  # 避免所有不确定的原子操作，保证得到一样的结果
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)


def create_model(num_classes, num_classes_2=0, in_channel=3, base_c=32, model='unet'):
    if model == 'unet':
        model = UNet(in_channels=in_channel, num_classes=num_classes, num_classes_2=num_classes_2, base_c=base_c)
        print('create unet model successfully')
    elif model in ['resnet50', 'resnet101']:
        layer_num = int(model.split('resnet')[-1])
        model = ResenetUnet(num_classes=num_classes, layer_num=layer_num)
        print(f'create resnet unet {layer_num} model successfully')
    elif model == 'vmunet':
        from src.vmunet.vmunet import VMUNet
        model = VMUNet(num_classes=num_classes)
        print(f'create resnet unet VMUnet model successfully')
    elif model == 'convnext_unet':
        model = U_ConvNext(img_ch=in_channel, output_ch=num_classes, channels=base_c)
        print('create convnext unet model successfully.')
    elif model == 'mobilev3unet':
        model = MobileV3Unet(num_classes=num_classes)
        print('create mobilev3unet model successfully')
    elif model == 'vgg16unet':
        model = VGG16UNet(num_classes=num_classes)
        print('create vgg16unet model successfully')
    elif model == 'u2netlite':
        model = u2net.u2net_lite(num_classes)
        print('create u2net lite model successfully')
    elif model == 'u2netfull':
        model = u2net.u2net_full(num_classes)
        print('create u2net full model successfully')
    elif model == 'resnet34unet':
        model = resnet_unet.Resnet34(3, num_classes)
        print('create resnet unet model successfully')

    # model = HighResolutionNet(num_joints=num_classes, base_channel=base_c)
    return model
