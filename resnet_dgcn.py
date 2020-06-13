import torch.nn as nn
import torchvision

from dgcn import DGCNet
from torchvision.models.resnet import Bottleneck


def resnet_with_dgcn(use_spatial_gcn=True, use_feature_gcn=True, d=8):

    resnet = torchvision.models.segmentation.fcn_resnet50(pretrained=True)

    if use_spatial_gcn or use_feature_gcn:
        dgcnet = DGCNet(use_spatial_gcn, use_feature_gcn, d)

        resnet.backbone.layer4 = nn.Sequential(
            resnet.backbone.layer4[0],
            resnet.backbone.layer4[1],
            dgcnet,
            resnet.backbone.layer4[2],
        )
        resnet.classifier = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 20, kernel_size=1)
            )

    return resnet
    

import torch
if __name__ == "__main__":

    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    x = torch.ones((1, 3, 1024, 2048))
    resnet = resnet_with_dgcn(True, True)

    resnet.to(device)
    x = x.to(device)

    with torch.no_grad():
        print(resnet)
        print(resnet(x)['out'].shape)
