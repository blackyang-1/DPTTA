import os
import sys
import torch
import numpy as np
import torch.nn as nn
from lib import *

class RegressionBranch(nn.Module):
    def __init__(self):
        """Regression branch"""
        super(RegressionBranch, self).__init__()
        self.R_pool = C_B_E(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1)  #down sample to 8x8
        self.R_res_block_4 = R_res_block_v2(128, 128, stride=1)
        self.R_res_block_5 = R_res_block_v2(128, 128, stride=1)  #[B,128,15,15]
        self.R_res_block_6 = R_res_block_v1(128, [128, 32], change_dimension=True, stride=1)
        self.final_conv = C_B_E(32, 1, kernel_size=3, stride=1, padding=1, use_bn=False, activation=True)
        self.fc = nn.Linear(8 * 8, 64)

    def forward(self, x):
        R_pool = self.R_pool(x)
        R_res_block_4 = self.R_res_block_4(R_pool)
        R_res_block_5 = self.R_res_block_5(R_res_block_4)
        R_res_block_6 = self.R_res_block_6(R_res_block_5)
        Final_conv = self.final_conv(R_res_block_6)

        Final_conv = Final_conv.view(x.size(0), 1, -1)  
        predicted_sparse_code = self.fc(Final_conv)  #[B,1,64]
        predicted_sparse_code = predicted_sparse_code.squeeze(1)  #[B,64]

        return predicted_sparse_code


class Encoder(nn.Module):
    def __init__(self):
        """DTEMDNet encoder"""
        super(Encoder, self).__init__()
        self.dilated_conv1 = Dilated_Conv(1, 32, kernel_size=3, dilation=1, padding=1, use_bn=False, activation=True)
        self.dilated_conv2 = Dilated_Conv(32, 64, kernel_size=3, dilation=2, padding=2, use_bn=False, activation=True)
        self.res_block_1 = res_block_v1(64, [64, 128], change_dimension=True, stride=1)
        self.pool = nn.AdaptiveAvgPool2d((15, 15))
        self.res_block_2 = res_block_v2(128, 128, stride=1)
        self.res_block_3 = res_block_v2(128, 128, stride=1)
        
        self.regression_branch = RegressionBranch()

    def forward(self, x):
        dilated_conv1 = self.dilated_conv1(x)
        dilated_conv2 = self.dilated_conv2(dilated_conv1)
        res_block_1 = self.res_block_1(dilated_conv2)
        pool1 = self.pool(res_block_1)
        res_block_2 = self.res_block_2(pool1)
        res_block_3 = self.res_block_3(res_block_2)
        
        predicted_sparse_code = self.regression_branch(res_block_3)

        return [dilated_conv1, dilated_conv2, res_block_1, res_block_3], predicted_sparse_code

class Decoder(nn.Module):
    def __init__ (self, image_size=30):
        """DTEMDNet decoder"""
        super(Decoder,self).__init__()
        self.image_size = image_size
        self.res_block_4 = res_block_v2(128, 128, stride=1)
        self.res_block_5 = res_block_v2(128, 128, stride=1)
        self.up = B_R_Deconv2d(128, 128, k=3, stride=2)
        self.res_block_6 = res_block_v1(128, [128, 64], change_dimension=True, stride=1)
        self.dilated_conv3 = Dilated_Conv(64, 32, kernel_size=3, dilation=2, padding=2, use_bn=False, activation=True)
        self.dilated_conv4 = Dilated_Conv(32, 16, kernel_size=3, dilation=1, padding=1, use_bn=False, activation=True)
        self.final_conv = Dilated_Conv(32, 1, kernel_size=3, dilation=1, padding=1, use_bn=False, activation=False)

    def forward(self, encoder_outputs, predicted_sparse_code, dictionary):
        
        dilated_conv1, dilated_conv2, res_block_1, res_block_3 = encoder_outputs
        #Dictionary reconstruction process
        DRSample = torch.matmul(dictionary, predicted_sparse_code.T) #[900,64] X [64,B] -> [900, B]
        DRSample = DRSample.T  # transposition -> [batch_size, 900]

        batch_size = DRSample.size(0)  
        image_size = self.image_size
        DRSample_2D = torch.zeros((batch_size, image_size, image_size), device=DRSample.device)
        for batch_idx in range(batch_size):
            array = DRSample[batch_idx].reshape(image_size, image_size)  # transpose -> [30x30]
            for num in range(1, int(image_size / 2) + 1):
                array[(num * 2) - 1] = torch.flip(array[(num * 2) - 1], dims=[0])
            DRSample_2D[batch_idx] = array

        DRSample_2D = DRSample_2D.unsqueeze(1)  #[B,1,30,30]
        DRSample_2D = DRSample_2D.repeat(1, 16, 1, 1)
        
        
        #decoder
        res_block_4 = self.res_block_4(res_block_3)
        res_block_5 = self.res_block_5(res_block_4)
        up = self.up(res_block_5,target_size=dilated_conv1.shape[2:]) + res_block_1
        res_block_6 = self.res_block_6(up) + dilated_conv2
        dilated_conv3 = self.dilated_conv3(res_block_6) + dilated_conv1
        dilated_conv4 = self.dilated_conv4(dilated_conv3)
        combined_features = torch.cat((dilated_conv4, DRSample_2D), dim=1)  # Concatenate with the dictionary result ->(B,32,30,30)
        output = self.final_conv(combined_features)

        return output

class DTEMDNet(nn.Module):
    def __init__(self, image_size=30):
        super(DTEMDNet, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(image_size)

    def forward(self, x, dictionary):
        encoder_outputs , predicted_sparse_code = self.encoder(x)        
        output = self.decoder(encoder_outputs, predicted_sparse_code,dictionary)
        return output, predicted_sparse_code