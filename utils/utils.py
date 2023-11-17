import os
import numpy as np
from skimage import measure


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def dice_score(prediction, target):
    smooth = 1e-5
    num_classes = target.size(0)
    prediction = prediction.view(num_classes, -1)
    target = target.view(num_classes, -1)

    intersection = (prediction * target)

    dice = (2. * intersection.sum(1) + smooth) / (prediction.sum(1) + target.sum(1) + smooth)

    return dice



def dice_score_batch(prediction, target):
    smooth = 1e-5
    batchsize = target.size(0)
    num_classes = target.size(1)
    prediction = prediction.view(batchsize, num_classes, -1)
    target = target.view(batchsize, num_classes, -1)
    intersection = (prediction * target)
    dice = (2. * intersection.sum(2) + smooth) / (prediction.sum(2) + target.sum(2) + smooth)
    return dice


def measure_img(o_img, t_num=1):
    p_img=np.zeros_like(o_img)
    testa1 = measure.label(o_img.astype("bool"))
    props = measure.regionprops(testa1)
    numPix = []
    for ia in range(len(props)):
        numPix += [props[ia].area]
    for i in range(0, t_num):
        index = numPix.index(max(numPix)) + 1
        p_img[testa1 == index]=o_img[testa1 == index]
        numPix[index-1]=0
    return p_img
