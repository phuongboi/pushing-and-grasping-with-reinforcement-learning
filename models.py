#!/usr/bin/env python

from collections import OrderedDict
import numpy as np
from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import matplotlib.pyplot as plt
import time
import re

class OneHeadNetwork(nn.Module):

    def __init__(self): # , snapshot=None
        super(OneHeadNetwork, self).__init__()

        # Initialize network trunks with DenseNet pre-trained on ImageNet
        self.img_backbone = torchvision.models.mobilenet_v2(pretrained=True)
        #print(self.img_backbone)
        #self.depth_backbone = torchvision.models.mobilenet_v2(pretrained=True)


        # Construct network branches for pushing and grasping
        self.prediction_head = nn.Sequential(OrderedDict([
            ('head-norm0', nn.BatchNorm2d(1280)),
            ('head-relu0', nn.ReLU(inplace=True)),
            ('head-conv0', nn.Conv2d(1280, 16, kernel_size=1, stride=1, bias=False)),
            # ('push-norm1', nn.BatchNorm2d(64)),
            # ('push-relu1', nn.ReLU(inplace=True)),
            # ('push-conv1', nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=False))
            ('head-upsample0', nn.Upsample(scale_factor=16, mode='bilinear'))
        ]))

        # Initialize network weights
        for m in self.named_modules():
            if 'head-' in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    nn.init.kaiming_normal_(m[1].weight.data)
                elif isinstance(m[1], nn.BatchNorm2d):
                    m[1].weight.data.fill_(1)
                    m[1].bias.data.zero_()




    def forward(self, input_img_data, input_depth_data):
        #print(input_img_data.shape)
        img_feat = self.img_backbone.features(input_img_data)
        # depth_feat = self.depth_backbone.features(input_depth_data)
        # interm_feat = torch.cat((img_feat, depth_feat), dim=1)
        output = self.prediction_head(img_feat)
        #print(output.shape)


        return output


class TwoHeadGraspNetwork(nn.Module):

    def __init__(self): # , snapshot=None
        super(TwoHeadGraspNetwork, self).__init__()

        # Initialize network trunks with DenseNet pre-trained on ImageNet
        self.model = torchvision.models.mobilenet_v2(pretrained=True)
        #self.feature_extractor = torch.nn.Sequential(*list(model.children())[:-2])
        #print(self.model)
        #print(self.img_backbone)
        #self.depth_backbone = torchvision.models.mobilenet_v2(pretrained=True)
        def weight_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.)

        # Construct head for orientation
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.orientation_head = nn.Sequential(nn.Linear(in_features=1280, out_features=16, bias=True))
        self.orientation_head.apply(weight_init)
        #  Construct head for location
        self.location_head = nn.Sequential(nn.Conv2d(1280, 64, kernel_size=1, stride=1, bias=False),
                                            nn.BatchNorm2d(64),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=False),
                                            nn.Upsample(scale_factor=16, mode='bilinear'))
        self.location_head.apply(weight_init)


    def forward(self, input_img_data, input_depth_data):
        #print(input_img_data.shapt
        img_feat = self.model.features(input_img_data)

        img_feat_view = self.avg_pool(img_feat).reshape(img_feat.size(0), -1)
        # depth_feat = self.depth_backbone.features(input_depth_data)
        # interm_feat = torch.cat((img_feat, depth_feat), dim=1)
        out_orient = self.orientation_head(img_feat_view)
        out_loc = self.location_head(img_feat)
        out_loc = out_loc.squeeze(1)
        #print(out_orient.shape, out_loc.shape)


        return out_orient, out_loc
