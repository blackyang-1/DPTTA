import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib import *

class DnCNN(nn.Module):
    def __init__(self):
        super(DnCNN,self).__init__()
        self.conv1 = C_B_R(1, 64, kernel_size=3, stride=1, padding=1, use_bn=False, activation=True)
        self.conv_layers = nn.ModuleList([C_B_R(64, 64, kernel_size=3, stride=1, padding=1, use_bn=True, activation=True) for _ in range(13)])
        self.conv_fianl = C_B_R(64, 1, kernel_size=3, stride= 1, padding=1, use_bn=False, activation=False)
    
    def forward(self,x):
        x = self.conv1(x)
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        x = self.conv_fianl(x)
        return x