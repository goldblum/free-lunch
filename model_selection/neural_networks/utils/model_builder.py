import torchvision.models as models
from .models_cifar import *
from .dataset_utils import get_num_classes

def build_model(model, dataset_name, num_classes = None, pretrained = False, imagenet_resize = False):
    '''
    pretrained is only available for torchvision imagenet models.
    '''
    if num_classes==None:
        num_classes=get_num_classes(dataset_name)
    
    if (dataset_name=='CIFAR10' or dataset_name=='CIFAR100' or dataset_name=='SVHN') and not imagenet_resize:
        if model == 'mobilenet_v2':
            net = MobileNetV2(num_classes=num_classes)
        elif model == 'wide_resnet_28x10_new':
            net = WideResNet_New(28, num_classes, widen_factor=10, dropRate = 0.3)
        elif model == 'wide_resnet_28x10':
            net = Wide_ResNet(28, 10, 0.3, num_classes)
        elif model == 'wide_resnet_28x20':
            net = Wide_ResNet(28, 20, 0.3, num_classes)
        elif model == 'wide_resnet_34x10': #NO DROPOUT, THIS IS WHAT PEOPLE USE FOR ADVERSARIAL STUFF
            net = Wide_ResNet(34, 10, 0.0, num_classes)
        elif model == 'wide_resnet_40x10':
            net = Wide_ResNet(40, 10, 0.3, num_classes)
        elif model == 'wide_resnet_40x20':
            net = Wide_ResNet(40, 20, 0.3, num_classes)
        elif model == 'preactresnet18':
            net = preactresnet18(num_classes=num_classes)
        elif model == 'preactresnet50':
            net = preactresnet50(num_classes=num_classes)
        elif model == 'resnet8':
            net = resnet8(num_classes=num_classes)
        elif model == 'resnet14':
            net = resnet14(num_classes=num_classes)
        elif model == 'resnet20':
            net = resnet20(num_classes=num_classes)
        elif model == 'resnet32':
            net = resnet32(num_classes=num_classes)
        elif model == 'resnet44':
            net = resnet44(num_classes=num_classes)
        elif model == 'resnet56':
            net = resnet56(num_classes=num_classes)
        elif model == 'resnet110':
            net = resnet110(num_classes=num_classes)
        elif model == 'resnet8x4':
            net = resnet8x4(num_classes=num_classes)
        elif model == 'resnet8x4_double':
            net = resnet8x4_double(num_classes=num_classes)
        elif model == 'resnet32x4':
            net = resnet32x4(num_classes=num_classes)               
        elif model == 'resnet10':
            net = ResNet10(num_classes=num_classes)
        elif model == 'resnet18':
            net = ResNet18(num_classes=num_classes)
        elif model == 'resnet34':
            net = ResNet34(num_classes=num_classes)
        elif model == 'resnet50':
            net = ResNet50(num_classes=num_classes)
        elif model == 'resnet101':
            net = ResNet101(num_classes=num_classes)
        elif model == 'resnet152':
            net = ResNet152(num_classes=num_classes)
        elif model == 'seresnet18':
            net = seresnet18(num_classes=num_classes)
        elif model == 'seresnet34':
            net = seresnet50(num_classes=num_classes)
        elif model == 'seresnet50':
            net = seresnet50(num_classes=num_classes)
        elif model == 'seresnet101':
            net = seresnet50(num_classes=num_classes)
        elif model == 'seresnet152':
            net = seresnet50(num_classes=num_classes)
        elif model == 'resnext101_32x4d':
            net = resnext101(num_classes=num_classes)
        elif model == 'resnext152_32x4d':
            net = resnext152(num_classes=num_classes)
        elif model == 'resnext50_32x4d':
            net = resnext50(num_classes=num_classes)
        elif model == 'mobilenet_v1':
            net = mobilenet(alpha=1, num_classes=num_classes)
        elif model == 'shufflenet_v1':
            net = shufflenet(num_classes=num_classes)
        elif model == 'shufflenet_v2':
            net = shufflenetv2(num_classes=num_classes)
        elif model == 'attention56':
            net = attention56(num_classes=num_classes)
        elif model == 'attention92':
            net = attention92(num_classes=num_classes)
        elif model == 'densenet121':
            net = densenet121(num_classes=num_classes)
        elif model == 'densenet169':
            net = densenet169(num_classes=num_classes)
        elif model == 'densenet201':
            net = densenet201(num_classes=num_classes)
        elif model == 'densenet161':
            net = densenet161(num_classes=num_classes)
        elif model == 'densenetbc_40':
            net = densenet_BC_cifar(40, 12, num_classes=num_classes)
        elif model == 'densenetbc_100':
            net = densenet_BC_cifar(100, 12, num_classes=num_classes)
        elif model == 'densenetbc_250':
            net = densenet_BC_cifar(250, 24, num_classes=num_classes)
        elif model == 'densenetbc_190':
            net = densenet_BC_cifar(190, 40, num_classes=num_classes)
        elif model == 'densenetbc_efficient_40':
            net = densenet_BC_cifar_efficient(40, 12, num_classes=num_classes)
        elif model == 'densenetbc_efficient_100':
            net = densenet_BC_cifar_efficient(100, 12, num_classes=num_classes)
        elif model == 'densenetbc_efficient_250':
            net = densenet_BC_cifar_efficient(250, 24, num_classes=num_classes)
        elif model == 'densenetbc_efficient_190':
            net = densenet_BC_cifar_efficient(190, 40, num_classes=num_classes)
        elif model == 'googlenet':
            net = googlenet(num_classes=num_classes)
        elif model == 'inception_v3':
            net = inceptionv3(num_classes=num_classes)
        elif model == 'inception_v4':
            net = inceptionv4(num_classes=num_classes)
        elif model == 'inception_resnet_v2':
            net = inception_resnet_v2(num_classes=num_classes)
        elif model == 'nasnet':
            net = nasnet(num_classes=num_classes)
        elif model == 'rir':
            net = resnet_in_resnet(num_classes=num_classes)
        elif model == 'squeezenet':
            net = squeezenet(num_classes=num_classes)
        elif model == 'stochastic_depth_resnet18':
            net = stochastic_depth_resnet18(num_classes=num_classes)
        elif model == 'stochastic_depth_resnet34':
            net = stochastic_depth_resnet34(num_classes=num_classes)
        elif model == 'stochastic_depth_resnet50':
            net = stochastic_depth_resnet50(num_classes=num_classes)
        elif model == 'stochastic_depth_resnet101':
            net = stochastic_depth_resnet101(num_classes=num_classes)
        elif model == 'stochastic_depth_resnet152':
            net = stochastic_depth_resnet152(num_classes=num_classes)
        elif model == 'vgg8_bn':
            net = vgg8_bn(num_classes=num_classes)
        elif model == 'vgg11_bn':
            net = vgg11_bn(num_classes=num_classes)
        elif model == 'vgg13_bn':
            net = vgg13_bn(num_classes=num_classes)
        elif model == 'vgg16_bn':
            net = vgg16_bn(num_classes=num_classes)
        elif model == 'vgg19_bn':
            net = vgg19_bn(num_classes=num_classes)
        elif model == 'vgg8':
            net = vgg8(num_classes=num_classes)
        elif model == 'vgg11':
            net = vgg11(num_classes=num_classes)
        elif model == 'vgg13':
            net = vgg13(num_classes=num_classes)
        elif model == 'vgg16':
            net = vgg16(num_classes=num_classes)
        elif model == 'vgg19':
            net = vgg19(num_classes=num_classes)
        elif model == 'xception':
            net = xception(num_classes=num_classes)
        elif model == 'dpn26':
            net = DPN26(num_classes=num_classes)
        elif model == 'dpn92':
            net = DPN92(num_classes=num_classes)
        elif model == 'dla':
            net = DLA(num_classes=num_classes)
    elif dataset_name=='ImageNet' or dataset_name=='TinyImageNet' or imagenet_resize:
        '''
        Options include: resnet18, resnet34, resnet50, resnet101, resnet152,
        densenet121, densenet169, densenet161, densenet201, inception_v3, mobilenet_v2,
        shufflenet_v2_x0_5, shufflenet_v2_x1_0, shufflenet_v2_x1_5, shufflenet_v2_x2_0,
        resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2,
        mnasnet0_5, mnasnet0_75, mnasnet1_0, mnasnet1_3, mobilenet_v3_large,
        mobilenet_v3_small, squeezenet1_0, squeezenet1_1
        '''
        if pretrained:
            if model == 'swin_b':
                weights = models.Swin_B_Weights.IMAGENET1K_V1
            elif model == 'vit_b_16':
                weights = models.ViT_B_16_Weights.IMAGENET1K_V1
            elif model == 'googlenet':
                weights = models.GoogLeNet_Weights.IMAGENET1K_V1
            else:
                raise Exception('No weights specified for this model')
            return models.__dict__[model](weights = weights)
        else:
            if model == 'googlenet':
                return models.GoogLeNet(num_classes=num_classes, aux_logits=False)
            return models.__dict__[model](num_classes=num_classes)
    return net
