import torch
from torch import nn
from torch.nn.functional import one_hot


class DSCLoss(nn.Module):
    def __init__(self, num_classes=2, inter_weight=0.5, intra_weights=None, device='cuda', is_3d=False):
        super(DSCLoss, self).__init__()
        if intra_weights is not None:
            intra_weights = torch.tensor(intra_weights).to(device)
        self.ce_loss = nn.CrossEntropyLoss(weight=intra_weights)
        self.num_classes = num_classes
        self.intra_weights = intra_weights
        self.inter_weight = inter_weight
        self.device = device
        self.is_3d = is_3d

    def dice_loss(self, prediction, target, weights=None):
        """Calculating the dice loss
        Args:
            prediction = predicted image
            target = Targeted image
        Output:
            dice_loss"""
        smooth = 1e-5

        prediction = torch.softmax(prediction, dim=1)
        batchsize = target.size(0)
        num_classes = target.size(1)
        prediction = prediction.view(batchsize, num_classes, -1)
        target = target.view(batchsize, num_classes, -1)

        intersection = (prediction * target)

        dice = (2. * intersection.sum(2) + smooth) / (prediction.sum(2) + target.sum(2) + smooth)
        dice_loss = 1 - dice.sum(0) / batchsize
        if weights is not None:
            weighted_dice_loss = dice_loss * weights
            return weighted_dice_loss.mean()
        return dice_loss.mean()

    def forward(self, pred, label):
        """Calculating the loss and metrics
            Args:
                prediction = predicted image
                target = Targeted image
                metrics = Metrics printed
                bce_weight = 0.5 (default)
            Output:
                loss : dice loss of the epoch """
        cel = self.ce_loss(pred, label)
        if self.is_3d:
            label_onehot = one_hot(label, num_classes=self.num_classes).permute(0, 4, 1, 2, 3).contiguous()
        else:
            label_onehot = one_hot(label, num_classes=self.num_classes).permute(0, 3, 1, 2).contiguous()
        dicel = self.dice_loss(pred, label_onehot, self.intra_weights)
        loss = cel * (1 - self.inter_weight) + dicel * self.inter_weight
        return loss
