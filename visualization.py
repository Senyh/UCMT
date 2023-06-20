import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
import argparse
from skimage import io
from skimage import color
import torch
import torch.nn.functional as F
from torchvision.transforms import Normalize
from torch.utils.data import DataLoader
from data.dataset import ISICDataset
from models import deeplabv3
from utils.utils import ensure_dir
sep = '\\' if sys.platform[:3] == 'win' else '/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_args(known=False):
    parser = argparse.ArgumentParser(description='PyTorch Implementation')
    parser.add_argument('--project', type=str, default=os.path.dirname(os.path.realpath(__file__)) + '/runs/exp_UCMT', help='project path for saving results')
    parser.add_argument('--backbone', type=str, default='UNet', choices=['DeepLabv3p', 'UNet'], help='segmentation backbone')
    parser.add_argument('--data_path', type=str, default='YOUR_DATA_PATH', help='path to the data')
    parser.add_argument('--is_cutmix', type=bool, default=False, help='cut mix')
    parser.add_argument('--labeled_percentage', type=float, default=0.1, help='the percentage of labeled data')
    parser.add_argument('--image_size', type=int, default=256, help='the size of images for training and testing')
    parser.add_argument('--batch_size', type=int, default=1, help='number of inputs per batch')
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


def vis(is_debug=True):
    args = get_args()
    # Project Saving Path
    project_path = args.project + '_{}_label_{}/'.format(args.backbone, args.labeled_percentage)
    save_path = project_path + 'visualization/'
    ensure_dir(save_path)
    # Load Data
    test_dataloader, length = get_data(args=args)
    iters = len(test_dataloader)
    iter_test_dataloader = iter(test_dataloader)
    if is_debug:
        pbar = range(100)
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
    un_norm = Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                        std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
    with torch.no_grad():
        for idx in pbar:
            image, label = iter_test_dataloader.next()
            image, label = image.to(device), label.to(device)
            pred = model(image)['out']
            B, C, H, W = label.shape
            pred = F.interpolate(pred, size=[H, W], mode='bilinear', align_corners=False)
            image = F.interpolate(image, size=[H, W], mode='bilinear', align_corners=False)
            pred = torch.softmax(pred, dim=1)
            pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy() * 255. / 3.
            label = label.squeeze(0).squeeze(0).long().cpu().numpy() * 255. / 3.
            image = un_norm(image)
            image = image.squeeze(0).cpu().numpy() * 255.
            image = image.transpose([1, 2, 0])
            io.imsave(save_path + str(idx) + '_img.png', image.astype('uint8'))
            io.imsave(save_path + str(idx) + '_lbl.png',
                      color.label2rgb((label).astype('uint8'),
                                      colors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
            io.imsave(save_path + str(idx) + '_prd.png',
                      color.label2rgb((pred).astype('uint8'),
                                      colors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
            print('itr/itrs: {}/{}'.format(idx + 1, len(pbar)))

    print('EVAL FINISHED!')


if __name__ == '__main__':
    vis()
