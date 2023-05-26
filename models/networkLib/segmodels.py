import time
from models.networkLib._utils import IntermediateLayerGetter
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
from models.networkLib.backbone import resnet
from models.networkLib._deeplabv3 import DeepLabHead, DeepLabV3, DeepLabHeadV3Plus
from models.networkLib._fcn import FCN, FCNHead

__all__ = ['fcn_resnet50', 'fcn_resnet101', 'deeplabv3_resnet50', 'deeplabv3_resnet101']


model_urls = {
    'fcn_resnet50_coco': 'https://download.pytorch.org/models/fcn_resnet50_coco-1167a1af.pth',
    'fcn_resnet101_coco': 'https://download.pytorch.org/models/fcn_resnet101_coco-7ecb50ca.pth',
    'deeplabv3_resnet50_coco': 'https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth',
    'deeplabv3_resnet101_coco': 'https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth',
    'deeplabv3p_resnet50_coco': 'https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth',
}


def _segm_resnet(name, backbone_name, in_channels, num_classes, aux, pretrained_backbone=True):
    backbone = resnet.__dict__[backbone_name](
        in_channels=in_channels,
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=[False, True, True])

    return_layers = {'layer4': 'out', 'layer1': 'low_level'}
    # if aux:
    #     return_layers['layer3'] = 'aux'
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = None
    # if aux:
    #     inplanes = 1024
    #     aux_classifier = FCNHead(inplanes, num_classes)

    model_map = {
        'deeplabv3': (DeepLabHead, DeepLabV3),
        'deeplabv3p': (DeepLabHeadV3Plus, DeepLabV3),
        'fcn': (FCNHead, FCN),
    }
    inplanes = 2048
    classifier = model_map[name][0](inplanes, num_classes)
    base_model = model_map[name][1]

    model = base_model(backbone, classifier, aux_classifier)
    return model


def _load_model(arch_type, backbone, pretrained, progress, in_channels, num_classes, aux_loss, **kwargs):
    if pretrained:
        aux_loss = True
    model = _segm_resnet(arch_type, backbone, in_channels, num_classes, aux_loss, **kwargs)
    if pretrained:
        t_start = time.time()
        arch = arch_type + '_' + backbone + '_coco'
        model_url = model_urls[arch]
        if model_url is None:
            raise NotImplementedError('pretrained {} is not supported as of now'.format(arch))
        else:
            # get the pretrained model dict
            state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
            t_ioend = time.time()
            model_dict = model.state_dict()
            # ini the corresponding layer
            i = 0
            j = 0
            for k, v in state_dict.items():
                if k in model_dict.keys():
                    if v.size() == model_dict[k].size():
                        model_dict[k] = state_dict[k]
                        i = i + 1
                    j = j + 1
            print('total weight is', j)
            print('using weight is', i)

            model.load_state_dict(model_dict, strict=False)
            ckpt_keys = set(state_dict.keys())
            own_keys = set(model.state_dict().keys())
            missing_keys = own_keys - ckpt_keys
            unexpected_keys = ckpt_keys - own_keys

            del state_dict
            t_end = time.time()
            print("Load segmentation model, Time usage:\n\tIO: {}, initialize parameters: {}".format(
                t_ioend - t_start, t_end - t_ioend))
    return model


def fcn_resnet50(pretrained=False, progress=True,
                 in_channels=1, num_classes=21, aux_loss=None, **kwargs):
    """Constructs a Fully-Convolutional Network model with a ResNet-50 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _load_model('fcn', 'resnet50', pretrained, progress, in_channels, num_classes, aux_loss, **kwargs)


def fcn_resnet101(pretrained=False, progress=True,
                  in_channels=1, num_classes=21, aux_loss=None, **kwargs):
    """Constructs a Fully-Convolutional Network model with a ResNet-101 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _load_model('fcn', 'resnet101', pretrained, progress, in_channels, num_classes, aux_loss, **kwargs)


def deeplabv3_resnet50(pretrained=False, progress=True,
                       in_channels=1, num_classes=21, aux_loss=None, **kwargs):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _load_model('deeplabv3', 'resnet50', pretrained, progress, in_channels, num_classes, aux_loss, **kwargs)


def deeplabv3_resnet101(pretrained=False, progress=True,
                        in_channels=1, num_classes=21, aux_loss=None, **kwargs):
    """Constructs a DeepLabV3 model with a ResNet-101 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _load_model('deeplabv3', 'resnet101', pretrained, progress, in_channels, num_classes, aux_loss, **kwargs)


def deeplabv3p_resnet50(pretrained=False, progress=True,
                       in_channels=1, num_classes=21, aux_loss=None, **kwargs):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _load_model('deeplabv3p', 'resnet50', pretrained, progress, in_channels, num_classes, aux_loss, **kwargs)


if __name__ == '__main__':
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = torch.randn([2, 1, 224, 224]).to(device)
    resnet = deeplabv3p_resnet50(in_channels=1, num_classes=4, pretrained=True).to(device)
    out = resnet(tensor)
    print(out['out'].shape)

