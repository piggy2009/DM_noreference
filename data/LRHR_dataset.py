from io import BytesIO
import lmdb
from PIL import Image
from torch.utils.data import Dataset
import random
import data.util as Util
import numpy as np
import cv2
from matplotlib import pyplot as plt
# from model.utils import categories
def color_filter(image):
    gray_SR = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_SR_gray = np.zeros_like(image)
    img_SR_gray[:, :, 0] = gray_SR
    img_SR_gray[:, :, 1] = gray_SR
    img_SR_gray[:, :, 2] = gray_SR

    result = image.copy()

    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # lower boundary RED color range values; Hue (0 - 10)
    lower1 = np.array([0, 0, 20])
    upper1 = np.array([40, 255, 255])

    # upper boundary RED color range values; Hue (160 - 180)
    lower2 = np.array([125, 0, 20])
    upper2 = np.array([179, 255, 255])

    lower_mask = cv2.inRange(image, lower1, upper1)
    upper_mask = cv2.inRange(image, lower2, upper2)

    full_mask = lower_mask + upper_mask
    # print(np.unique(full_mask))
    neg_full_mask = 255 - full_mask

    result = cv2.bitwise_and(result, result, mask=full_mask)
    result_neg = cv2.bitwise_and(img_SR_gray, img_SR_gray, mask=neg_full_mask)
    return result_neg + result[:, :, ::-1]

class LRHRDataset2(Dataset):
    def __init__(self, dataroot, datatype, l_resolution=16, r_resolution=128, split='train', data_len=-1, need_LR=False, label_path=None):
        self.datatype = datatype
        self.l_res = l_resolution
        self.r_res = r_resolution
        self.data_len = data_len
        self.need_LR = need_LR
        self.split = split
        self.need_label = False
        if label_path is not None:
            labels = {}
            for line in open(label_path):
                image_name, label = line.split(' ')
                labels[image_name] = int(label.strip())
            self.labels = labels
            self.need_label = True

        if datatype == 'img':
            self.sr_path = Util.get_paths_from_images(
                '{}/sr_{}_{}_gray'.format(dataroot, l_resolution, r_resolution))
            self.style_path = Util.get_paths_from_images(
                '{}/style_{}'.format(dataroot, r_resolution))
            self.dataset_len = len(self.sr_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        else:
            raise NotImplementedError(
                'data_type [{:s}] is not recognized.'.format(datatype))

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        index_style = np.random.randint(0, self.data_len)
        img_SR = Image.open(self.sr_path[index]).convert("RGB")
        img_style = Image.open(self.style_path[index_style]).convert("RGB").resize((256, 256))
        label = self.labels[self.sr_path[index]]

        [img_SR, img_style] = Util.transform_augment(
            [img_SR, img_style], split=self.split, min_max=(-1, 1))
        return {'SR': img_SR, 'style': img_style, 'label': label, 'Index': index}

