import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data.dataset import ISICDataset
from models import deeplabv3
from utils.utils import dice_score_batch
sep = '\\' if sys.platform[:3] == 'win' else '/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_args(known=False):
    parser = argparse.ArgumentParser(description='PyTorch Implementation')
    parser.add_argument('--project', type=str, default=os.path.dirname(os.path.realpath(__file__)) + '/runs/UCMT', help='project path for saving results')
    parser.add_argument('--backbone', type=str, default='UNet', choices=['DeepLabv3p', 'UNet'], help='segmentation backbone')
    parser.add_argument('--data_path', type=str, default='YOUR_DATA_PATH', help='path to the data')
    parser.add_argument('--is_cutmix', type=bool, default=False, help='cut mix')
    parser.add_argument('--labeled_percentage', type=float, default=0.1, help='the percentage of labeled data')
    parser.add_argument('--image_size', type=int, default=256, help='the size of images for training and testing')
    parser.add_argument('--batch_size', type=int, default=8, help='number of inputs per batch')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers to use for dataloader')
    parser.add_argument('--in_channels', type=int, default=3, help='input channels')
    parser.add_argument('--num_classes', type=int, default=2, help='number of target categories')
    parser.add_argument('--model_weights', type=str, default='best.pth', help='model weights')
    args = parser.parse_known_args()[0] if known else parser.parse_args()
    return args


def get_data(args):
    test_set = ISICDataset(image_path=args.data_path, stage='test', image_size=args.image_size, is_augmentation=False)
    test_dataloder = DataLoader(dataset=test_set, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    return test_dataloder, len(test_set)


def load_model(model_weights, in_channels, num_classes, backbone):
    model = deeplabv3.__dict__[backbone](in_channels=in_channels, out_channels=num_classes).to(device)
    print('#parameters:', sum(param.numel() for param in model.parameters()))
    model.load_state_dict(torch.load(model_weights))
    return model


def eval(is_debug=False):
    args = get_args()
    # Project Saving Path
    project_path = args.project + '_{}_label_{}/'.format(args.backbone, args.labeled_percentage)
    # Load Data
    test_dataloader, length = get_data(args=args)
    iters = len(test_dataloader)
    iter_test_dataloader = iter(test_dataloader)
    if is_debug:
        pbar = range(10)
        length = 10 * args.batch_size
    else:
        pbar = range(iters)
    # Load model
    weights_path = project_path + 'weights/' + args.model_weights
    model = load_model(model_weights=weights_path, in_channels=args.in_channels, num_classes=args.num_classes, backbone=args.backbone)
    model.eval()
    ############################
    # Evaluation
    ############################
    print('start evaluation')
    results = {i: [] for i in range(args.num_classes)}
    with torch.no_grad():
        for idx in pbar:
            image, label = next(iter_test_dataloader)
            image, label = image.to(device), label.to(device)
            pred = model(image)['out']
            B, C, H, W = label.shape
            pred = F.interpolate(pred, size=[H, W], mode='bilinear', align_corners=False)
            pred = torch.softmax(pred, dim=1)
            pred = torch.argmax(pred, dim=1)
            label = label.squeeze(1).long()
            label_onehot = torch.nn.functional.one_hot(label, num_classes=args.num_classes).permute(0, 3, 1, 2).contiguous()
            pred_onehot = torch.nn.functional.one_hot(pred, num_classes=args.num_classes).permute(0, 3, 1, 2).contiguous()
            dices = dice_score_batch(prediction=pred_onehot, target=label_onehot).cpu().numpy()
            for b in range(len(dices)):
                for i in range(args.num_classes):
                    results[i].append(dices[b][i])
            print('itr/itrs: {}/{}, label: {}, pred: {}'.format(idx + 1, len(pbar), label.shape, pred.shape))
    # save results
    data_frame = pd.DataFrame(
        data={i: results[i] for i in range(args.num_classes)},
        index=range(1, length + 1))
    data_frame.to_csv(project_path + '/' + 'evaluation.csv', index_label='Index')
    result = data_frame.values
    avg_score = np.mean(result, axis=0)
    with open(project_path+'/performance.txt', 'w') as f:
        f.writelines('metric is {} \n'.format(avg_score[1:]))
    print('AVG Score:', avg_score[1:])
    print('EVAL FINISHED!')


if __name__ == '__main__':
    eval()
