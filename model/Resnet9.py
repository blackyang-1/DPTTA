import torch
import torch.nn as nn
import torch.nn.functional as F
from lib import *

class ResNet9(nn.Module):
    def __init__(self):
        super(ResNet9, self).__init__()
        
        self.dilated_conv1 = res_block_v1(1, [1, 64], change_dimension=True, stride=1)
        self.dilated_conv2 = res_block_v2(64, 64, stride=1)

        self.res_block_1 = res_block_v1(64, [64, 64], change_dimension=True, stride=1)
        self.res_block_2 = res_block_v2(64, 64, stride=1)
        self.res_block_3 = res_block_v2(64, 64, stride=1)
        self.res_block_4 = res_block_v2(64, 64, stride=1)
        self.res_block_5 = res_block_v2(64, 64, stride=1)

        self.dilated_conv3 = res_block_v2(64, 64, stride=1)
        self.dilated_conv4 = res_block_v1(64, [64, 1], change_dimension=True, stride=1)

    def forward(self, x):
        x = self.dilated_conv1(x)  
        x = self.dilated_conv2(x)  
        x = self.res_block_1(x)    
        x = self.res_block_2(x)   
        x = self.res_block_3(x)    
        x = self.res_block_4(x)   
        x = self.res_block_5(x)    

        x = self.dilated_conv3(x)  
        x = self.dilated_conv4(x)  
        
        return x
