import torch.nn as nn
import torch.nn.functional as F

#code by black-y 2025.10.15 16:02
#DTEMDNET model basic_layers

#Trunk block
#Note: B:BN, R:ReLU, C:Conv

#Upsample block using conv
class B_R_Deconv2d(nn.Module):
    def __init__(self, inp_c, oup_c, k, stride=1, use_bn=True, activation=True):
        super(B_R_Deconv2d, self).__init__()
        self.use_bn = use_bn
        if self.use_bn:
            self.bn = nn.BatchNorm2d(inp_c)
        else:
            self.bn = nn.Identity()
        if activation:
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Identity() 

        self.upsample = nn.Upsample(scale_factor=stride, mode='bilinear', align_corners=False) if stride > 1 else nn.Identity()
        self.conv = nn.Conv2d(inp_c, oup_c, kernel_size=k, padding=k // 2)

    def forward(self, x, target_size=None):
        if self.use_bn:
            x = self.bn(x)
        x = self.activation(x)
        x = self.upsample(x)  
        x = self.conv(x)  
        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)  # 调整到目标尺寸
        return x
    
#TEMDnet basic block
class C_B_R(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_bn=True, activation=True):
        super(C_B_R, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True)
        self.use_bn = use_bn 
        self.activation = activation 
        if self.use_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        if self.activation:
            self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.activation:
            x = self.relu(x)
        return x

#Resblock version 1
class res_block_v1(nn.Module):
    def __init__(self, in_channels, out_channels, change_dimension=False, stride=1):
        super(res_block_v1, self).__init__()
        
        self.change_dimension = change_dimension
        if change_dimension:
            self.shortcut_conv = C_B_R(in_channels, out_channels[1], kernel_size=1, stride=stride, padding=0, activation=False)
        else:
            self.shortcut_conv = nn.Identity()
            
        self.conv1 = C_B_R(in_channels, out_channels[0], kernel_size=1, stride=stride, padding=0)
        self.conv2 = C_B_R(out_channels[0], out_channels[0], kernel_size=3, stride=1, padding=1)
        self.conv3 = C_B_R(out_channels[0], out_channels[1], kernel_size=1, stride=1, padding=0, activation=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = self.shortcut_conv(x)
        x = self.conv1(x) #1X1
        x = self.conv2(x) #3X3
        x = self.conv3(x) #1X1
        x = x + shortcut  
        return self.relu(x)

#Resblock version 2
class res_block_v2(nn.Module):
    def __init__(self,in_channels, out_channels, stride=1):
        super(res_block_v2,self).__init__()
        self.shortcut_conv = nn.Identity()
        self.conv1 = C_B_R(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = C_B_R(out_channels, out_channels, kernel_size=3, stride=1, padding=1, activation=False)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self,x):
        shortcut = self.shortcut_conv(x)
        x = self.conv1(x)#3X3
        x = self.conv2(x)#3X3
        x = x + shortcut
        return self.relu(x)

class Dilated_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, padding=1, use_bn=True, activation=True):
        super(Dilated_Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding, dilation=dilation, bias=True)
        self.use_bn = use_bn
        self.activation = activation
        if self.use_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        if self.activation:
            self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.activation:
            x = self.relu(x)
        return x

#Regression branch 2D block
#Note: R:regression, E:ELU 
class C_B_E(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_bn=True, activation=True):
        super(C_B_E, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True)
        self.use_bn = use_bn
        self.activation = activation
        if self.use_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        if self.activation:
            self.act = nn.ELU(alpha=6.0, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.activation:
            x = self.act(x)
        return x

class R_res_block_v1(nn.Module):
    def __init__(self, in_channels, out_channels, change_dimension=False, stride=1):
        super(R_res_block_v1, self).__init__()
        self.change_dimension = change_dimension
        if change_dimension:
            self.shortcut_conv = C_B_E(in_channels, out_channels[1], kernel_size=1, stride=stride, padding=0, activation=False)
        else:
            self.shortcut_conv = nn.Identity()
        self.conv1 = C_B_E(in_channels, out_channels[0], kernel_size=1, stride=stride, padding=0)
        self.conv2 = C_B_E(out_channels[0], out_channels[0], kernel_size=3, stride=1, padding=1)
        self.conv3 = C_B_E(out_channels[0], out_channels[1], kernel_size=1, stride=1, padding=0, activation=False)
        self.elu = nn.ELU(alpha=6.0, inplace=True)

    def forward(self, x):
        shortcut = self.shortcut_conv(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x + shortcut
        return self.elu(x)

class R_res_block_v2(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(R_res_block_v2, self).__init__()
        self.shortcut_conv = nn.Identity()
        self.conv1 = C_B_E(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = C_B_E(out_channels, out_channels, kernel_size=3, stride=1, padding=1, activation=False)
        self.elu = nn.ELU(alpha=6.0, inplace=True)

    def forward(self, x):
        shortcut = self.shortcut_conv(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + shortcut
        return self.elu(x)

class R_Dilated_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, padding=1, use_bn=True, activation=True):
        super(R_Dilated_Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding, dilation=dilation, bias=True)
        self.use_bn = use_bn
        self.activation = activation
        if self.use_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        if self.activation:
            self.act = nn.ELU(alpha=6.0, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.activation:
            x = self.act(x)
        return x



