import torch
import torch.nn as nn
import torch.nn.functional as F

class CoordinateAttention(nn.Module):
    """
    坐标注意力机制：专门针对细长条状特征（如赤潮、藻类线条）
    比普通的SE Attention更适合分割任务，因为它保留了位置信息。
    """
    def __init__(self, in_channels, reduction=32):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1)) # (H, 1)
        self.pool_w = nn.AdaptiveAvgPool2d((1, None)) # (1, W)

        mip = max(8, in_channels // reduction)

        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish() # Hardswish通常比ReLU效果更好

        self.conv_h = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        
        # 1. 分别沿H和W方向进行池化，捕捉长距离依赖
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        # 2. 拼接特征并进行卷积变换
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        # 3. 生成注意力权重
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        # 4. 加权原特征
        out = identity * a_h * a_w
        return out

class MS_CSM(nn.Module):
    """
    Multi-Scale Coordinate Strip Module
    结合了多尺度空洞卷积和坐标注意力
    """
    def __init__(self, in_channels, out_channels):
        super(MS_CSM, self).__init__()
        mid_channels = out_channels // 2
        
        # 分支1：保留局部细节（针对小斑点）
        self.branch_local = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        # 分支2：扩大感受野（针对大片区域），使用空洞卷积
        self.branch_dilated = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        # 融合后的 1x1 卷积
        self.conv_cat = nn.Sequential(
            nn.Conv2d(mid_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 核心：坐标注意力，强化细长特征
        self.ca = CoordinateAttention(out_channels)

    def forward(self, x):
        x1 = self.branch_local(x)
        x2 = self.branch_dilated(x)
        
        # 特征融合
        x_cat = torch.cat([x1, x2], dim=1)
        x_out = self.conv_cat(x_cat)
        
        # 应用注意力
        x_out = self.ca(x_out)
        
        # 残差连接（如果输入输出通道一致，建议加上残差）
        if x.shape[1] == x_out.shape[1]:
            x_out = x_out + x
            
        return x_out