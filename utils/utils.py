import os


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