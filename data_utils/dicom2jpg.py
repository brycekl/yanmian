import os

import matplotlib.pyplot as plt
import numpy as np
import pydicom
from PIL import Image


def dicom_jpg(dcm_path, output_path):
    data = pydicom.read_file(dcm_path)
    plt.imsave(data.pixel_array[:, :, 0], output_path)


def read_dir(root):
    paths = os.listdir(root)
    files = []
    for path in paths:
        dir = os.path.join(root, path)
        if os.path.isdir(dir):
            files.extend(read_dir(dir))
        elif os.path.isfile(dir) and dir.endswith('.dcm'):
            files.append(root)
    return files


if __name__ == '__main__':
    # 下面是将对应的dicom格式的图片转成jpg、G:\Pycharm\yanmian\data\abnormal\颜面异常\input\新21\5010797
    input_path = './abnormal/datas/ss'
    dcm_paths = read_dir(input_path)
    dcm_paths = set(dcm_paths)
    for dcm_path in dcm_paths:
        for dcm_name in [path for path in os.listdir(dcm_path) if path.endswith('.dcm') and not path.endswith('OBGYN.dcm')]:
            data = pydicom.read_file(os.path.join(dcm_path, dcm_name))
            output_path = dcm_path.replace('ss', 'ss_o')
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            if hasattr(data, 'pixel_array'):
                output_data = np.zeros_like(data.pixel_array)
                for i in range(3):
                    output_data[:, :, i] = data.pixel_array[:, :, 0]
                out_img = Image.fromarray(output_data)
                out_img.save(os.path.join(output_path, dcm_name.replace('.dcm', '.jpg')))
                # cv2.imwrite(os.path.join(output_path, dcm_name + '.png'), output_data)
