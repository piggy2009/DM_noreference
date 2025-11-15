import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

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
    return result_neg + result

root = '/home/ty/code/DM_noreference/dataset/RUIE_val_16_256/sr_16_256'
save_path = '/home/ty/code/DM_noreference/dataset/RUIE_val_16_256/sr_16_256_gray'
if not os.path.exists(save_path):
    os.makedirs(save_path)

images = os.listdir(root)

for image in images:
    img = cv2.imread(os.path.join(root, image))
    # print(img.shape)
    output = color_filter(img)
    cv2.imwrite(os.path.join(save_path, image), output)


# cv2.imshow('mask', full_mask)
# cv2.imshow('result', result)

# cv2.waitKey(0)
# cv2.destroyAllWindows()