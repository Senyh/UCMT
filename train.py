import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
import time
import argparse
import copy
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from easydict import EasyDict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, Subset, random_split
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Normalize, ToPILImage
from data.dataset import ISICDataset
from models import deeplabv3
from utils.loss_functions import DSCLoss
from utils.logger import logger as logging
from utils.utils import *
from utils.mask_generator import BoxMaskGenerator, AddMaskParamsToBatch, SegCollate
from utils.ramps import sigmoid_rampup
from utils.torch_utils import seed_torch
from utils.model_init import init_weight
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_args(known=False):
    parser = argparse.ArgumentParser(description='PyTorch Implementation')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--project', type=str, default=os.path.dirname(os.path.realpath(__file__)) + '/runs/UCMT', help='project path for saving results')
    parser.add_argument('--backbone', type=str, default='DeepLabv3p', choices=['DeepLabv3p', 'UNet'], help='segmentation backbone')
    parser.add_argument('--data_path', type=str, default='YOUR_DATA_PATH', help='path to the data')
    parser.add_argument('--image_size', type=int, default=256, help='the size of images for training and testing')
    parser.add_argument('--labeled_percentage', type=float, default=0.05, help='the percentage of labeled data')
    parser.add_argument('--is_cutmix', type=bool, default=False, help='cut mix')
    parser.add_argument('--mix_prob', type=float, default=0.5, help='probability for amplitude mix')
    parser.add_argument('--topk', type=float, default=2, help='top k')
    parser.add_argument('--num_epochs', type=int, default=25, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='number of inputs per batch')
    parser.add_argument('--num_workers', type=int, default=2, help='number of workers to use for dataloader')
    parser.add_argument('--in_channels', type=int, default=3, help='input channels')
    parser.add_argument('--num_classes', type=int, default=2, help='number of target categories')
    parser.add_argument('--pretrained', type=bool, default=True, help='using pretrained weights')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--intra_weights', type=list, default=[1., 1.], help='inter classes weighted coefficients in the loss function')
    parser.add_argument('--inter_weight', type=float, default=1., help='inter losses weighted coefficients in the loss function')
    parser.add_argument('--log_freq', type=float, default=10, help='logging frequency of metrics accord to the current iteration')
    args = parser.parse_known_args()[0] if known else parser.parse_args()
    return args


def get_data(args):
    train_set = ISICDataset(image_path=args.data_path, stage='train', image_size=args.image_size, is_augmentation=True)
    labeled_train_set, unlabeled_train_set = random_split(train_set, [int(len(train_set) * args.labeled_percentage),
                                                       len(train_set) - int(len(train_set) * args.labeled_percentage)])
    print('before:', len(labeled_train_set), len(train_set))
    # repeat the labeled set to have a equal length with the unlabeled set (dataset)
    labeled_ratio = len(train_set) // len(labeled_train_set)
    labeled_train_set = ConcatDataset([labeled_train_set for i in range(labeled_ratio)])
    labeled_train_set = ConcatDataset([labeled_train_set,
                                       Subset(labeled_train_set, range(len(train_set) - len(labeled_train_set)))])
    assert len(labeled_train_set) == len(train_set)
    print('after:', len(labeled_train_set), len(train_set))
    train_labeled_dataloder = DataLoader(dataset=labeled_train_set, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    train_unlabeled_dataloder = DataLoader(dataset=train_set, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    if args.is_cutmix:
        mask_generator = BoxMaskGenerator(prop_range=(0.25, 0.5),
                                          n_boxes=3,
                                          random_aspect_ratio=True,
                                          prop_by_area=True,
                                          within_bounds=True,
                                          invert=True)

        add_mask_params_to_batch = AddMaskParamsToBatch(mask_generator)
        mask_collate_fn = SegCollate(batch_aug_fn=add_mask_params_to_batch)
        aux_dataloder = DataLoader(dataset=labeled_train_set, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, pin_memory=True, collate_fn=mask_collate_fn)
        return train_labeled_dataloder, train_unlabeled_dataloder, aux_dataloder
    return train_labeled_dataloder, train_unlabeled_dataloder


def main(is_debug=False):
    args = get_args()
    seed_torch(args.seed)
    # Project Saving Path
    project_path = args.project + '_{}_label_{}/'.format(args.backbone, args.labeled_percentage)
    ensure_dir(project_path)
    save_path = project_path + 'weights/'
    ensure_dir(save_path)

    # Tensorboard & Statistics Results & Logger
    tb_dir = project_path + '/tensorboard{}'.format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
    writer = SummaryWriter(tb_dir)
    metrics = EasyDict()
    metrics.train_loss = []
    metrics.train_loss2 = []
    metrics.val_loss = []
    logger = logging(project_path + 'train_val.log')
    logger.info('PyTorch Version {}\n Experiment{}'.format(torch.__version__, project_path))

    # Load Data
    if args.is_cutmix:
        train_labeled_dataloader, train_unlabeled_dataloader, aux_loader = get_data(args=args)
    else:
        train_labeled_dataloader, train_unlabeled_dataloader = get_data(args=args)
    iters = len(train_labeled_dataloader)

    # Load Model & EMA
    student1 = deeplabv3.__dict__[args.backbone](in_channels=args.in_channels, out_channels=args.num_classes).to(device)
    student2 = deeplabv3.__dict__[args.backbone](in_channels=args.in_channels, out_channels=args.num_classes).to(device)
    teacher = deeplabv3.__dict__[args.backbone](in_channels=args.in_channels, out_channels=args.num_classes).to(device)
    init_weight(student1.net.classifier, nn.init.kaiming_normal_,
                nn.BatchNorm2d, 1e-5, 0.1,
                mode='fan_in', nonlinearity='relu')
    init_weight(student2.net.classifier, nn.init.kaiming_normal_,
                nn.BatchNorm2d, 1e-5, 0.1,
                mode='fan_in', nonlinearity='relu')
    init_weight(teacher.net.classifier, nn.init.kaiming_normal_,
                nn.BatchNorm2d, 1e-5, 0.1,
                mode='fan_in', nonlinearity='relu')
    teacher.detach_model()
    # best_model_wts = copy.deepcopy(teacher.state_dict())
    h, w = args.image_size // 16, args.image_size // 16
    s = h
    unfolds  = torch.nn.Unfold(kernel_size=(h, w), stride=s).to(device)
    folds = torch.nn.Fold(output_size=(args.image_size, args.image_size), kernel_size=(h, w), stride=s).to(device)
    best_epoch = 0
    best_loss = 100
    alpha = 0.1
    total_iters = iters * args.num_epochs

    # Criterion & Optimizer & LR Schedule
    criterion = DSCLoss(num_classes=args.num_classes, intra_weights=args.intra_weights, inter_weight=args.inter_weight, device=device)
    criterion_u = DSCLoss(num_classes=args.num_classes, intra_weights=args.intra_weights, inter_weight=args.inter_weight, device=device)
    criterion_c = DSCLoss(num_classes=args.num_classes, intra_weights=args.intra_weights, inter_weight=args.inter_weight, device=device)
    optimizer1 = optim.AdamW(student1.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
    optimizer2 = optim.AdamW(student2.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))

    # Train
    since = time.time()
    logger.info('start training')
    for epoch in range(1, args.num_epochs + 1):
        epoch_metrics = EasyDict()
        epoch_metrics.train_loss = []
        epoch_metrics.train_loss2 = []
        if is_debug:
            pbar = range(10)
        else:
            pbar = range(iters)
        iter_train_labeled_dataloader = iter(train_labeled_dataloader)
        iter_train_unlabeled_dataloader = iter(train_unlabeled_dataloader)

        ############################
        # Train
        ############################
        student1.train()
        student2.train()
        teacher.train()
        for idx in pbar:
            image, label, imageA1, imageA2 = next(iter_train_labeled_dataloader)
            image, label = image.to(device), label.to(device)
            imageA1, imageA2 = imageA1.to(device), imageA2.to(device)
            uimage, _, uimageA1, uimageA2 = next(iter_train_unlabeled_dataloader)
            uimage = uimage.to(device)
            uimageA1, uimageA2 = uimageA1.to(device), uimageA2.to(device)

            '''
            Step 1
            '''
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            ###########################
                # supervised path #
            ###########################
            pred1s = student1(image)
            pred2s = student2(image)
            preds = teacher(image)
            pred1 = pred1s['out']
            pred2 = pred2s['out']
            pred = preds['out']
            pred1_feature = torch.softmax(pred1, dim=1)
            pred2_feature = torch.softmax(pred2, dim=1)
            pred_feature = torch.softmax(pred, dim=1)
            loss1_sup = criterion(pred1, label.squeeze(1).long())
            loss2_sup = criterion(pred2, label.squeeze(1).long())
            loss_sup = loss1_sup + loss2_sup

            ###########################
                # unsupervised path #
            ###########################
            # Estimate the pseudo-labels
            pred1s_u = student1(uimageA1)
            pred2s_u = student2(uimageA2)
            preds_u = teacher(uimage)
            pred1_u = pred1s_u['out']
            pred2_u = pred2s_u['out']
            pred_u = preds_u['out']
            pred1_u_feature = torch.softmax(pred1_u, dim=1)
            pseudo1 = torch.argmax(pred1_u_feature, dim=1)
            pred2_u_feature = torch.softmax(pred2_u, dim=1)
            pseudo2 = torch.argmax(pred2_u_feature, dim=1)
            pred_u_feature = torch.softmax(pred_u, dim=1)
            pseudo = torch.argmax(pred_u_feature, dim=1)

            # CMT loss
            loss1_cmt = criterion_c(pred1_u, pseudo.detach())
            loss2_cmt = criterion_c(pred2_u, pseudo.detach())
            loss_cmt = (loss1_cmt + loss2_cmt) * 0.5
            
            # CPS loss
            loss1_u = criterion_u(pred1_u, pseudo2.detach())
            loss2_u = criterion_u(pred2_u, pseudo1.detach())
            loss_cps = (loss1_u + loss2_u) * 0.5
            loss_u = (loss_cps + loss_cmt) * alpha
            lambda_ = sigmoid_rampup(current=idx + len(pbar) * (epoch-1), rampup_length=len(pbar)*5)
            loss = loss_sup + lambda_ * loss_u
            loss.backward()
            optimizer1.step()
            optimizer2.step()
            teacher.weighted_update(student1, student2, ema_decay=0.99, cur_step=idx + len(pbar) * (epoch-1))

            writer.add_scalar('train_sup_loss', loss_sup.item(), idx + len(pbar) * (epoch-1))
            writer.add_scalar('train_cps_loss', loss_cps.item(), idx + len(pbar) * (epoch-1))
            writer.add_scalar('train_cmt_loss', loss_cmt.item(), idx + len(pbar) * (epoch-1))
            writer.add_scalar('train_loss', loss.item(), idx + len(pbar) * (epoch-1))
            if idx % args.log_freq == 0:
                logger.info("Train1: Epoch/Epochs {}/{}\t"
                            "iter/iters {}/{}\t"
                            "loss {:.3f}, loss_sup {:.3f}, loss_cps {:.3f}, loss_cmt {:.3f}, lambda {}".format(epoch, args.num_epochs, idx, len(pbar),
                                                                                  loss.item(), loss_sup.item(), loss_cps.item(), loss_cmt.item(), lambda_))
            epoch_metrics.train_loss.append(loss.item())

            '''
            Step 2
            '''
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            topk = args.topk
            ###########################
                # supervised path #
            ###########################
            # Estimate the uncertainty map
            with torch.no_grad():
                uncertainty_map11 = torch.mean(torch.stack([pred1_feature, pred_feature]), dim=0)
                uncertainty_map11 = -1.0 * torch.sum(uncertainty_map11*torch.log(uncertainty_map11 + 1e-6), dim=1, keepdim=True)
                uncertainty_map22 = torch.mean(torch.stack([pred2_feature, pred_feature]), dim=0)
                uncertainty_map22 = -1.0 * torch.sum(uncertainty_map22*torch.log(uncertainty_map22 + 1e-6), dim=1, keepdim=True)
                
                B, C = image.shape[0], image.shape[1]
                # for student 1
                x11 = unfolds(uncertainty_map11)  # B x C*kernel_size[0]*kernel_size[1] x L
                x11 = x11.view(B, 1, h, w, -1)  # B x C x h x w x L
                x11_mean = torch.mean(x11, dim=(1, 2, 3))  # B x L
                _, x11_max_index = torch.sort(x11_mean, dim=1, descending=True)  # B x L B x L
                # for student 2
                x22 = unfolds(uncertainty_map22)  # B x C*kernel_size[0]*kernel_size[1] x L
                x22 = x22.view(B, 1, h, w, -1)  # B x C x h x w x L
                x22_mean = torch.mean(x22, dim=(1, 2, 3))  # B x L
                _, x22_max_index = torch.sort(x22_mean, dim=1, descending=True)  # B x L B x L
                img_unfold = unfolds(imageA1).view(B, C, h, w, -1)  # B x C x h x w x L
                lab_unfold = unfolds(label.float()).view(B, 1, h, w, -1)  # B x C x h x w x L
                for i in range(B):
                    img_unfold[i, :, :, :, x11_max_index[i, :topk]] = img_unfold[i, :, :, :, x22_max_index[i, -topk:]]
                    img_unfold[i, :, :, :, x22_max_index[i, :topk]] = img_unfold[i, :, :, :, x11_max_index[i, -topk:]]
                    lab_unfold[i, :, :, :, x11_max_index[i, :topk]] = lab_unfold[i, :, :, :, x22_max_index[i, -topk:]]
                    lab_unfold[i, :, :, :, x22_max_index[i, :topk]] = lab_unfold[i, :, :, :, x11_max_index[i, -topk:]]
                image2 = folds(img_unfold.view(B, C*h*w, -1))
                label2 = folds(lab_unfold.view(B, 1*h*w, -1))
            pred1s = student1(image2)
            pred2s = student2(image2)
            pred1 = pred1s['out']
            pred2 = pred2s['out']
            loss1_sup = criterion(pred1, label2.squeeze(1).long())
            loss2_sup = criterion(pred2, label2.squeeze(1).long())
            loss_sup = loss1_sup + loss2_sup

            ###########################
                # unsupervised path #
            ###########################
            # Estimate the uncertainty map
            with torch.no_grad():
                uncertainty_map1 = torch.mean(torch.stack([pred1_u_feature, pred_u_feature]), dim=0)
                uncertainty_map1 = -1.0 * torch.sum(uncertainty_map1*torch.log(uncertainty_map1 + 1e-6), dim=1, keepdim=True)
                uncertainty_map2 = torch.mean(torch.stack([pred2_u_feature, pred_u_feature]), dim=0)
                uncertainty_map2 = -1.0 * torch.sum(uncertainty_map2*torch.log(uncertainty_map2 + 1e-6), dim=1, keepdim=True)

                B, C = uimage.shape[0], uimage.shape[1]
                # for student 1
                x1 = unfolds(uncertainty_map1)  # B x C*kernel_size[0]*kernel_size[1] x L
                x1 = x1.view(B, 1, h, w, -1)  # B x C x h x w x L
                x1_mean = torch.mean(x1, dim=(1, 2, 3))  # B x L
                _, x1_max_index = torch.sort(x1_mean, dim=1, descending=True)  # B x L B x L
                # for student 2
                x2 = unfolds(uncertainty_map2)  # B x C*kernel_size[0]*kernel_size[1] x L
                x2 = x2.view(B, 1, h, w, -1)  # B x C x h x w x L
                x2_mean = torch.mean(x2, dim=(1, 2, 3))  # B x L
                _, x2_max_index = torch.sort(x2_mean, dim=1, descending=True)  # B x L B x L
                imgu_unfold = unfolds(uimageA1).view(B, C, h, w, -1)  # B x C x h x w x L
                pseudo_unfold = unfolds(pseudo.unsqueeze(1).float()).view(B, 1, h, w, -1)  # B x C x h x w x
                for i in range(B):
                    imgu_unfold[i, :, :, :, x1_max_index[i, :topk]] = imgu_unfold[i, :, :, :, x2_max_index[i, -topk:]]
                    imgu_unfold[i, :, :, :, x2_max_index[i, :topk]] = imgu_unfold[i, :, :, :, x1_max_index[i, -topk:]]
                    pseudo_unfold[i, :, :, :, x1_max_index[i, :topk]] = pseudo_unfold[i, :, :, :, x2_max_index[i, -topk:]]
                    pseudo_unfold[i, :, :, :, x2_max_index[i, :topk]] = pseudo_unfold[i, :, :, :, x1_max_index[i, -topk:]]
                uimage2 = folds(imgu_unfold.view(B, C*h*w, -1))
                pseudo = folds(pseudo_unfold.view(B, 1*h*w, -1)).squeeze(1).long()
  
            # Re-Estimate the pseudo-labels on the new uimages
            pred1_u = student1(uimage2)['out']
            pred2_u = student2(uimage2)['out']

            pseudo1 = torch.softmax(pred1_u, dim=1)
            pseudo1 = torch.argmax(pseudo1, dim=1)
            pseudo2 = torch.softmax(pred2_u, dim=1)
            pseudo2 = torch.argmax(pseudo2, dim=1)
            # CMT loss
            loss1_cmt = criterion_c(pred1_u, pseudo.detach())
            loss2_cmt = criterion_c(pred2_u, pseudo.detach())
            loss_cmt = (loss1_cmt + loss2_cmt) * 0.5
            
            # CPS loss
            loss1_u = criterion_u(pred1_u, pseudo2.detach())
            loss2_u = criterion_u(pred2_u, pseudo1.detach())
            loss_cps = (loss1_u + loss2_u) * 0.5
            loss_u = (loss_cps + loss_cmt) * alpha
            lambda_ = sigmoid_rampup(current=idx + len(pbar) * (epoch-1), rampup_length=len(pbar)*5)
            loss = loss_sup + lambda_ * loss_u
            loss.backward()
            optimizer1.step()
            optimizer2.step()

            teacher.weighted_update(student1, student2, ema_decay=0.99, cur_step=idx + len(pbar) * (epoch-1))

            writer.add_scalar('train_sup_loss', loss_sup.item(), idx + len(pbar) * (epoch-1))
            writer.add_scalar('train_cps_loss', loss_cps.item(), idx + len(pbar) * (epoch-1))
            writer.add_scalar('train_cmt_loss', loss_cmt.item(), idx + len(pbar) * (epoch-1))
            writer.add_scalar('train_loss', loss.item(), idx + len(pbar) * (epoch-1))
            if idx % args.log_freq == 0:
                logger.info("Train2: Epoch/Epochs {}/{}\t"
                            "iter/iters {}/{}\t"
                            "loss {:.3f}, loss_sup {:.3f}, loss_cps {:.3f}, loss_cmt {:.3f}, lambda {}".format(epoch, args.num_epochs, idx, len(pbar),
                                                                                  loss.item(), loss_sup.item(), loss_cps.item(), loss_cmt.item(), lambda_))
            epoch_metrics.train_loss2.append(loss.item())
        
        metrics.train_loss.append(np.mean(epoch_metrics.train_loss))
        metrics.train_loss2.append(np.mean(epoch_metrics.train_loss2))
        logger.info("Average: Epoch/Epoches {}/{}\t"
                    "train1 epoch loss {:.3f}\t"
                    "train2 epoch loss {:.3f}\n".format(epoch, args.num_epochs,
                     np.mean(epoch_metrics.train_loss), np.mean(epoch_metrics.train_loss2), ))
        if np.mean(epoch_metrics.train_loss2) <= best_loss:
            best_loss = np.mean(epoch_metrics.train_loss2)
            torch.save(teacher.state_dict(), save_path + 'best.pth'.format(best_epoch))
        torch.save(teacher.state_dict(), save_path + 'last.pth'.format(best_epoch))
    ############################
    # Save Metrics
    ############################
    data_frame = pd.DataFrame(
        data={'loss': metrics.train_loss,
              'loss2': metrics.train_loss2},
        index=range(1, args.num_epochs + 1))
    data_frame.to_csv(project_path + 'train_loss.csv', index_label='Epoch')
    plt.figure()
    plt.title("Loss")
    plt.plot(metrics.train_loss, label="Train")
    plt.plot(metrics.train_loss2, label="Train2")
    plt.xlabel("epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(project_path + 'train_loss.png')

    time_elapsed = time.time() - since
    logger.info('Training completed in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    logger.info(project_path)
    logger.info('TRAINING FINISHED!')


if __name__ == '__main__':
    main()
