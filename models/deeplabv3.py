import torch
import torch.nn as nn
from models.vnet import VNetorg
from models.networkLib.segmodels import deeplabv3_resnet50, deeplabv3p_resnet50


class DeepLabv3(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, pretrained=False):
        super(DeepLabv3, self).__init__()
        self.net = deeplabv3_resnet50(in_channels=in_channels, num_classes=out_channels, pretrained=pretrained)

    def forward(self, x):
        x = self.net(x)
        return x

    def detach_model(self):
        for param in self.parameters():
            param.detach_()

    def ema_update(self, student, ema_decay=0.99, cur_step=None):
        if cur_step is not None:
            ema_decay = min(1 - 1 / (cur_step + 1), ema_decay)
        for t_param, s_param in zip(self.parameters(), student.parameters()):
            t_param.data.mul_(ema_decay).add_(1 - ema_decay, s_param.data)

    def weighted_update(self, teacher1, teacher2, coefficient=0.99, ema_decay=0.99, cur_step=None):
        if cur_step is not None:
            ema_decay = min(1 - 1 / (cur_step + 1), ema_decay)
        for t_param, t1_param, t2_param in zip(self.parameters(), teacher1.parameters(), teacher2.parameters()):
            t_param.data.mul_(ema_decay).add_(coefficient*(1-ema_decay), t1_param.data).add_((1-coefficient)*(1-ema_decay), t2_param.data)


class DeepLabv3p(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, pretrained=False):
        super(DeepLabv3p, self).__init__()
        self.net = deeplabv3p_resnet50(in_channels=in_channels, num_classes=out_channels, pretrained=pretrained)

    def forward(self, x):
        x = self.net(x)
        return x

    def detach_model(self):
        for param in self.parameters():
            param.detach_()

    def ema_update(self, student, ema_decay, cur_step=None):
        if cur_step is not None:
            ema_decay = min(1 - 1 / (cur_step + 1), ema_decay)
        for t_param, s_param in zip(self.parameters(), student.parameters()):
            t_param.data.mul_(ema_decay).add_(1 - ema_decay, s_param.data)

    def weighted_update(self, teacher1, teacher2, coefficient=0.99, ema_decay=0.99, cur_step=None):
        if cur_step is not None:
            ema_decay = min(1 - 1 / (cur_step + 1), ema_decay)
        for t_param, t1_param, t2_param in zip(self.parameters(), teacher1.parameters(), teacher2.parameters()):
            t_param.data.mul_(ema_decay).add_(coefficient*(1-ema_decay), t1_param.data).add_((1-coefficient)*(1-ema_decay), t2_param.data)


class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""
    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv_conv(x)

class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""
    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""
    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p, mode_upsampling=1):
        super(UpBlock, self).__init__()
        self.mode_upsampling = mode_upsampling
        if mode_upsampling==0:
            self.up = nn.ConvTranspose2d(in_channels1, in_channels2, kernel_size=2, stride=2)
        elif mode_upsampling==1:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        elif mode_upsampling==2:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode='nearest')
        elif mode_upsampling==3:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.mode_upsampling != 0:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5)
        self.in_conv = ConvBlock(self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(self.ft_chns[3], self.ft_chns[4], self.dropout[4])

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]

class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.up_type = self.params['up_type']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0, mode_upsampling=self.up_type)
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0, mode_upsampling=self.up_type)
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0, mode_upsampling=self.up_type)
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0, mode_upsampling=self.up_type)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=3, padding=1)

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        output = self.out_conv(x)
        return output

    
class UNetorg(nn.Module):
    def __init__(self, in_channels=1, num_classes=1):
        super(UNetorg, self).__init__()

        params1 = {'in_chns': in_channels,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': num_classes,
                  'up_type': 1,
                  'acti_func': 'relu'}

        self.backbone = Encoder(params1)
        self.classifier = Decoder(params1)
        self.kaiming_normal_init_weight()

    def forward(self, x):
        feature = self.backbone(x)
        output = self.classifier(feature)
        return {'out': output}
    
    def kaiming_normal_init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        self.net = UNetorg(in_channels=in_channels, num_classes=out_channels)

    def forward(self, x):
        x = self.net(x)
        return x

    def detach_model(self):
        for param in self.parameters():
            param.detach_()

    def ema_update(self, student, ema_decay=0.99, cur_step=None):
        if cur_step is not None:
            ema_decay = min(1 - 1 / (cur_step + 1), ema_decay)
        for t_param, s_param in zip(self.parameters(), student.parameters()):
            t_param.data.mul_(ema_decay).add_(1 - ema_decay, s_param.data)
    
    def weighted_update(self, teacher1, teacher2, coefficient=0.99, ema_decay=0.99, cur_step=None):
        if cur_step is not None:
            ema_decay = min(1 - 1 / (cur_step + 1), ema_decay)
        for t_param, t1_param, t2_param in zip(self.parameters(), teacher1.parameters(), teacher2.parameters()):
            t_param.data.mul_(ema_decay).add_(coefficient*(1-ema_decay), t1_param.data).add_((1-coefficient)*(1-ema_decay), t2_param.data)



class VNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, normalization='batchnorm'):
        super(VNet, self).__init__()
        self.net = VNetorg(in_channels=in_channels, out_channels=out_channels, normalization=normalization)

    def forward(self, x, perturbation=False):
        x = self.net(x, perturbation)
        return x

    def detach_model(self):
        for param in self.parameters():
            param.detach_()

    def ema_update(self, student, ema_decay, cur_step=None):
        if cur_step is not None:
            ema_decay = min(1 - 1 / (cur_step + 1), ema_decay)
        for t_param, s_param in zip(self.parameters(), student.parameters()):
            t_param.data.mul_(ema_decay).add_(1 - ema_decay, s_param.data)
    
    def weighted_update(self, teacher1, teacher2, coefficient=0.99, ema_decay=0.99, cur_step=None):
        if cur_step is not None:
            ema_decay = min(1 - 1 / (cur_step + 1), ema_decay)
        for t_param, t1_param, t2_param in zip(self.parameters(), teacher1.parameters(), teacher2.parameters()):
            t_param.data.mul_(ema_decay).add_(coefficient*(1-ema_decay), t1_param.data).add_((1-coefficient)*(1-ema_decay), t2_param.data)

if __name__ == '__main__':
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = torch.randn([2, 1, 256, 256]).to(device)
    deeplab = DeepLabv3p(in_channels=1, out_channels=4, pretrained=True).to(device)
    result = deeplab(tensor)
    print('#parameters:', sum(param.numel() for param in deeplab.parameters()))
    print(result['out'].shape)