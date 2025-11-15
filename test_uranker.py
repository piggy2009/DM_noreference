import os
from PIL import Image
import pyiqa
import torch
# print(pyiqa.list_models())
import torchvision.transforms as transforms
import numpy as np
# Read a PIL image


# Define a transform to convert PIL
# image to a Torch tensor
transform = transforms.Compose([
    transforms.ToTensor()
])

# transform = transforms.PILToTensor()
# Convert the PIL image to Torch tensor

# path = '/media/ty/My Passport/compare/challenging-60/USUIR2'
path = './experiments_val/sr_ffhq_251112_105057/results'
images = os.listdir(path)
device = 'cuda:0'
uranker = pyiqa.create_metric('uranker', device=device)
Brisque = pyiqa.create_metric('brisque', device=device)

sum_uranker = 0.0
sum_Brisque = 0.0
count = 0

# score = uranker(path)
for img_name in images:
    count += 1

    image = transform(np.array(Image.open(os.path.join(path, img_name)).resize((128, 128)))).unsqueeze(0)

    score = uranker(image)
    score = score.data.cpu().numpy()
    sum_uranker = sum_uranker + score

    score = Brisque(image)
    score = score.data.cpu().numpy()
    sum_Brisque = sum_Brisque + score


print('# Validation # uranker: {0}'.format(sum_uranker / count))
print('# Validation # Brisque: {0}'.format(sum_Brisque / count))
