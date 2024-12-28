#PyTorch lib
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
#Tools lib
import numpy as np
import cv2
import random
import time
import os
from utils.common import toggle_grad

#Initialize VGG16 with pretrained weight on ImageNet
def vgg_init(device):
    vgg_model = torchvision.models.vgg16(pretrained = True).to(device)
    toggle_grad(vgg_model, False)
    return vgg_model

#Extract features from internal layers for perceptual loss
class Vgg(nn.Module):
    def __init__(self, vgg_model):
        super(Vgg, self).__init__()
        self.vgg_layers = vgg_model.features
        self.layer_name_mapping = {
            '1': "relu1_1",
            '3': "relu1_2",
            '6': "relu2_1",
            '8': "relu2_2"
        }

    def forward(self, x):
        output = []
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output.append(x)
        return output