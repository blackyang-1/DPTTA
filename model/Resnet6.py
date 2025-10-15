import torch
import torch.nn as nn
import torch.nn.functional as F
from lib import *

class ResNet6(nn.Module):
    def __init__(self):
        super(ResNet6, self).__init__()

        self.dilated_conv1 = res_block_v1(1, [1, 32], change_dimension=True, stride=1)
        self.dilated_conv2 = res_block_v1(32, [32, 64], change_dimension=True, stride=1)

        self.res_block_1 = res_block_v1(64, [64, 128], change_dimension=True, stride=1)
        self.res_block_2 = res_block_v2(128, 128, stride=1)

        self.dilated_conv3 = res_block_v1(128, [64, 32], change_dimension=True, stride=1)
        self.dilated_conv4 = res_block_v1(32, [32, 1], change_dimension=True, stride=1)

    def forward(self, x):
        x = self.dilated_conv1(x)
        x = self.dilated_conv2(x)  
        x = self.res_block_1(x)    
        x = self.res_block_2(x)    
        x = self.dilated_conv3(x)  
        x = self.dilated_conv4(x)  
        
        return x