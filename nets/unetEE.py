import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.resnet import resnet50
from nets.vgg import VGG16

# =====================================================================
# 基础组件：FReLU 激活函数 (DEESBlock 需要)
# =====================================================================
class FReLU(nn.Module):
    def __init__(self, c1, k=3):  # ch_in, kernel
        super().__init__()
        # 使用 深度可分离卷积 DepthWise Separable Conv + BN 实现T(x)
        self.conv = nn.Conv2d(c1, c1, k, 1, 1, groups=c1, bias=False)
        self.bn = nn.BatchNorm2d(c1)

    def forward(self, x):
        # f(x)=max(x, T(x))
        return torch.max(x, self.bn(self.conv(x)))

# =====================================================================
# 【创新模块 1】DEESBlock: 差分边缘与弹性条带模块 (保留)
# 用于瓶颈层，处理深层长条状特征
# =====================================================================
class DEESBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DEESBlock, self).__init__()
        mid_channels = out_channels // 4 

        # --- 分支1：弹性水平条带 (1x9 Conv) ---
        self.strip_h = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=(1, 9), stride=1, padding=(0, 4), bias=False),
            nn.BatchNorm2d(mid_channels),
            FReLU(mid_channels)
        )

        # --- 分支2：弹性垂直条带 (9x1 Conv) ---
        self.strip_v = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=(9, 1), stride=1, padding=(4, 0), bias=False),
            nn.BatchNorm2d(mid_channels),
            FReLU(mid_channels)
        )

        # --- 分支3：中心差分边缘感知 (Center-Difference) ---
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False), 
            nn.BatchNorm2d(mid_channels),
            FReLU(mid_channels)
        )

        # --- 分支4：全局上下文锚点 (Global Context Anchor) ---
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            FReLU(mid_channels)
        )

        # --- 融合层 ---
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(mid_channels * 4, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            FReLU(out_channels) 
        )

    def forward(self, x):
        b, c, h, w = x.size()
        
        # 1. 四个分支前向传播
        feat_h = self.strip_h(x)
        feat_v = self.strip_v(x)
        
        local_avg = self.avg_pool(x)
        diff_feat = x - local_avg 
        feat_edge = self.edge_conv(diff_feat)

        global_feat = self.global_avg_pool(x)
        global_feat = self.global_conv(global_feat)
        global_feat = F.interpolate(global_feat, size=(h, w), mode='nearest')

        # 2. 拼接
        concat_feat = torch.cat([feat_h, feat_v, feat_edge, global_feat], dim=1)
        
        # 3. 融合与残差连接
        out = self.fusion_conv(concat_feat)
        
        if x.shape[1] == out.shape[1]:
            out = out + x
            
        return out

# =====================================================================
# UNet 解码块 (保持不变)
# =====================================================================
class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1  = nn.Conv2d(in_size, out_size, kernel_size = 3, padding = 1)
        self.conv2  = nn.Conv2d(out_size, out_size, kernel_size = 3, padding = 1)
        self.up     = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.relu   = nn.ReLU(inplace = True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs

# =====================================================================
# 最终的魔改 UNet 主体 (仅含 DEESBlock)
# =====================================================================
class Unet(nn.Module):
    def __init__(self, num_classes = 21, pretrained = False, backbone = 'resnet50'):
        super(Unet, self).__init__()
        if backbone == "resnet50":
            self.resnet = resnet50(pretrained = pretrained)
            # Bubbliiiing ResNet50 各层实际输出通道数
            feat1_c, feat2_c, feat3_c, feat4_c, feat5_c = 64, 256, 512, 1024, 2048
            bottleneck_channels = feat5_c
        else:
            raise ValueError('Unsupported backbone - `{}`, Use resnet50.'.format(backbone))
        
        out_filters = [64, 128, 256, 512]

        # 1. 瓶颈层创新模块：DEESBlock (输入2048，输出512)
        self.bottleneck_dees = DEESBlock(bottleneck_channels, out_filters[3])

        # 【已移除】跳跃连接上的注意力模块

        # 3. 解码器上采样 (通道数计算保持正确)
        # up4输入: feat4(1024) + bottleneck输出(512) = 1536. 输出: 512
        self.up_concat4 = unetUp(feat4_c + out_filters[3], out_filters[3])
        
        # up3输入: feat3(512) + up4输出(512) = 1024. 输出: 256
        # 【恢复原始连接】直接使用 feat3
        self.up_concat3 = unetUp(feat3_c + out_filters[3], out_filters[2])
        
        # up2输入: feat2(256) + up3输出(256) = 512. 输出: 128
        # 【恢复原始连接】直接使用 feat2
        self.up_concat2 = unetUp(feat2_c + out_filters[2], out_filters[1])
        
        # up1输入: feat1(64) + up2输出(128) = 192. 输出: 64
        self.up_concat1 = unetUp(feat1_c + out_filters[1], out_filters[0])

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

        self.backbone = backbone

    def forward(self, inputs):
        if self.backbone == "resnet50":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)

        # --- 创新点应用 ---
        # 1. 最深层：应用 DEESBlock
        feat5_enhanced = self.bottleneck_dees(feat5)

        # 【已移除】浅层跳跃连接上的注意力应用

        # --- 解码器路径 ---
        up4 = self.up_concat4(feat4, feat5_enhanced)
        # 【恢复原始连接】直接使用原始的 feat3 和 feat2 进行拼接
        up3 = self.up_concat3(feat3, up4) 
        up2 = self.up_concat2(feat2, up3) 
        up1 = self.up_concat1(feat1, up2)

        final = self.final(up1)
        
        return final

    def freeze_backbone(self):
        if self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True