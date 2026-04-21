import torch
import torch.nn as nn
import torch.nn.functional as F

# 假设 nets 包中有这两个文件，如果没有请替换为 torchvision 的实现
from nets.resnet import resnet50
from nets.vgg import VGG16

# =====================================================================
# 基础组件 1: FReLU (用于 DEESBlock)
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
# 基础组件 2: SELayer (用于 GSAGBlock)
# =====================================================================
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        reduced_channel = max(channel // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, reduced_channel, bias=False),
            nn.GELU(),
            nn.Linear(reduced_channel, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# =====================================================================
# 【创新模块 1】DEESBlock: 差分边缘与弹性条带模块
# 位置：放置在 Encoder 和 Decoder 之间的瓶颈层
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
        
        # 3. 融合
        out = self.fusion_conv(concat_feat)
        
        # 残差连接 (如果通道数一致)
        if x.shape[1] == out.shape[1]:
            out = out + x
            
        return out

# =====================================================================
# 【创新模块 2】GSAGBlock: 绿藻敏感注意力门控
# 位置：放置在 Decoder 的跳跃连接上
# =====================================================================
class GSAGBlock(nn.Module):
    def __init__(self, shallow_in_c, deep_in_c, out_c):
        super(GSAGBlock, self).__init__()
        
        # 1. 通道对齐卷积
        self.shallow_conv = nn.Sequential(
            nn.Conv2d(shallow_in_c, out_c, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.deep_conv = nn.Sequential(
            nn.Conv2d(deep_in_c, out_c, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_c)
        )

        # 2. 绿藻特征增强 (SE模块)
        self.se_layer = SELayer(out_c, reduction=8)

        # 3. 空间注意力图生成
        self.psi = nn.Sequential(
            nn.Conv2d(out_c, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.GELU()

    def forward(self, x_shallow, g_deep):
        # 对齐通道数
        x_s_aligned = self.shallow_conv(x_shallow)
        g_d_aligned = self.deep_conv(g_deep)

        # 上采样深层特征以匹配浅层特征的空间尺寸
        g_d_upsampled = F.interpolate(g_d_aligned, size=x_s_aligned.size()[2:], mode='bilinear', align_corners=True)
        
        # 特征相加融合
        fusion = x_s_aligned + g_d_upsampled
        fusion = self.relu(fusion)

        # 应用SE模块增强对绿藻通道的敏感性
        fusion_enhanced = self.se_layer(fusion)
        
        # 生成空间注意力系数图
        attention_map = self.psi(fusion_enhanced)
        
        # 将注意力图乘回原始的浅层特征
        return x_shallow * attention_map

# =====================================================================
# 标准组件：UNet 上采样模块
# =====================================================================
class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1  = nn.Conv2d(in_size, out_size, kernel_size = 3, padding = 1)
        self.conv2  = nn.Conv2d(out_size, out_size, kernel_size = 3, padding = 1)
        self.up     = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.relu   = nn.ReLU(inplace = True)

    def forward(self, inputs1, inputs2):
        # inputs1: 浅层特征 (GSAG 过滤后)
        # inputs2: 深层特征 (需要上采样)
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs

# =====================================================================
# 终极融合 Unet：包含 DEESBlock 和 GSAGBlock
# =====================================================================
class Unet(nn.Module):
    def __init__(self, num_classes = 21, pretrained = False, backbone = 'resnet50'):
        super(Unet, self).__init__()
        
        # -----------------------------------------------------------------
        # 1. 骨干网络设置
        # -----------------------------------------------------------------
        if backbone == 'vgg':
            self.vgg    = VGG16(pretrained = pretrained)
            # VGG 各层通道数: feat1=64, feat2=128, feat3=256, feat4=512, feat5=512
            feat1_c, feat2_c, feat3_c, feat4_c, feat5_c = 64, 128, 256, 512, 512
        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained = pretrained)
            # ResNet50 各层通道数: feat1=64, feat2=256, feat3=512, feat4=1024, feat5=2048
            feat1_c, feat2_c, feat3_c, feat4_c, feat5_c = 64, 256, 512, 1024, 2048
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        
        # 解码器各阶段的目标输出通道数
        out_filters = [64, 128, 256, 512]

        # -----------------------------------------------------------------
        # 2. 瓶颈层创新：DEESBlock
        # -----------------------------------------------------------------
        # 输入是 feat5，输出被强制压缩到 out_filters[3] (即512)，以便后续解码
        self.bottleneck_dees = DEESBlock(feat5_c, out_filters[3])
        # DEESBlock 输出后的通道数
        center_c = out_filters[3] 

        # -----------------------------------------------------------------
        # 3. 跳跃连接创新：GSAGBlock
        # -----------------------------------------------------------------
        # 只要有了 DEESBlock，feat5 变成了 center_c (512)
        
        # GSAG4: 过滤 feat4。指导信号来自 DEESBlock 增强后的 feat5
        self.gsag4 = GSAGBlock(shallow_in_c=feat4_c, deep_in_c=center_c, out_c=out_filters[3])
        
        # GSAG3: 过滤 feat3。指导信号来自 up4
        self.gsag3 = GSAGBlock(shallow_in_c=feat3_c, deep_in_c=out_filters[3], out_c=out_filters[2])

        # GSAG2: 过滤 feat2。指导信号来自 up3
        self.gsag2 = GSAGBlock(shallow_in_c=feat2_c, deep_in_c=out_filters[2], out_c=out_filters[1])

        # -----------------------------------------------------------------
        # 4. 解码器上采样模块
        # -----------------------------------------------------------------
        # 这里的输入通道数 = 浅层特征通道(GSAG后保持不变) + 深层特征通道(上采样后)
        
        # up4: feat4 + feat5_enhanced
        self.up_concat4 = unetUp(feat4_c + center_c, out_filters[3])
        
        # up3: feat3 + up4_out
        self.up_concat3 = unetUp(feat3_c + out_filters[3], out_filters[2])
        
        # up2: feat2 + up3_out
        self.up_concat2 = unetUp(feat2_c + out_filters[2], out_filters[1])
        
        # up1: feat1 + up2_out (第一层通常保留原始细节，不使用GSAG，直接拼接)
        self.up_concat1 = unetUp(feat1_c + out_filters[1], out_filters[0])

        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor = 2), 
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
            )
        else:
            self.up_conv = None

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)
        self.backbone = backbone

    def forward(self, inputs):
        # 1. 骨干网络提取特征
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
        elif self.backbone == "resnet50":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)

        # 2. 【DEESBlock】增强最深层特征
        # 这一步整合了多尺度条带信息和边缘差分信息
        feat5_enhanced = self.bottleneck_dees(feat5)

        # 3. 解码路径 (带 【GSAGBlock】)
        
        # Step 4: 
        # 用增强后的深层特征 指导 过滤 feat4
        feat4_gated = self.gsag4(x_shallow=feat4, g_deep=feat5_enhanced)
        up4 = self.up_concat4(feat4_gated, feat5_enhanced)

        # Step 3:
        # 用 up4 的输出 指导 过滤 feat3
        feat3_gated = self.gsag3(x_shallow=feat3, g_deep=up4)
        up3 = self.up_concat3(feat3_gated, up4)

        # Step 2:
        # 用 up3 的输出 指导 过滤 feat2
        feat2_gated = self.gsag2(x_shallow=feat2, g_deep=up3)
        up2 = self.up_concat2(feat2_gated, up3)

        # Step 1:
        # 最后一层直接拼接，保留最大分辨率细节
        up1 = self.up_concat1(feat1, up2)

        # 4. 最终输出
        if self.up_conv != None:
            up1 = self.up_conv(up1)

        final = self.final(up1)
        
        return final

    def freeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True