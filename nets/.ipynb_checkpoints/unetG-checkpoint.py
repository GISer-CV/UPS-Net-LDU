import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.resnet import resnet50
from nets.vgg import VGG16


# =====================================================================
# 基础组件：SELayer (Squeeze-and-Excitation, GSAGBlock 需要)
# 用于强调特征图中的“绿色/藻类”相关通道
# =====================================================================
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        # 如果通道数太少，至少保证reduction后为1
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
# 【创新模块】GSAGBlock: 绿藻敏感注意力门控
# 放置在跳跃连接上，利用深层语义信息过滤浅层纹理噪声
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
        # 在计算空间注意力之前，先强化对绿藻敏感的通道维度
        self.se_layer = SELayer(out_c, reduction=8)

        # 3. 空间注意力图生成
        self.psi = nn.Sequential(
            nn.Conv2d(out_c, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.GELU()

    def forward(self, x_shallow, g_deep):
        """
        x_shallow: 来自编码器的浅层特征 (高分辨率, 跳跃连接特征)
        g_deep: 来自解码器的深层特征 (低分辨率, 作为门控信号)
        """
        # 对齐通道数
        x_s_aligned = self.shallow_conv(x_shallow)
        g_d_aligned = self.deep_conv(g_deep)

        # 上采样深层特征以匹配浅层特征的空间尺寸
        # 使用 bilinear 插值确保平滑
        g_d_upsampled = F.interpolate(g_d_aligned, size=x_s_aligned.size()[2:], mode='bilinear', align_corners=True)
        
        # 特征相加融合
        fusion = x_s_aligned + g_d_upsampled
        fusion = self.relu(fusion)

        # 【关键点】应用SE模块增强对绿藻通道的敏感性
        fusion_enhanced = self.se_layer(fusion)
        
        # 生成空间注意力系数图 (范围 0-1)
        attention_map = self.psi(fusion_enhanced)
        
        # 将注意力图乘回原始的浅层特征
        # 注意：这里乘回的是原始输入的 x_shallow，不改变其通道数
        return x_shallow * attention_map


# =====================================================================
# 原有的 unetUp 模块 (保持不变)
# =====================================================================
class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1  = nn.Conv2d(in_size, out_size, kernel_size = 3, padding = 1)
        self.conv2  = nn.Conv2d(out_size, out_size, kernel_size = 3, padding = 1)
        self.up     = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.relu   = nn.ReLU(inplace = True)

    def forward(self, inputs1, inputs2):
        # inputs1: 浅层特征 (通常来自跳跃连接)
        # inputs2: 深层特征 (需要上采样)
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs

# =====================================================================
# 修改后的 Unet 主类 (添加了 GSAG 模块)
# =====================================================================
class Unet(nn.Module):
    def __init__(self, num_classes = 21, pretrained = False, backbone = 'vgg'):
        super(Unet, self).__init__()
        if backbone == 'vgg':
            self.vgg    = VGG16(pretrained = pretrained)
            # VGG encoder 各层输出通道数
            feat1_c, feat2_c, feat3_c, feat4_c, feat5_c = 64, 128, 256, 512, 512
            # unetUp 的输入通道数参考值 (用于后续计算)
            in_filters  = [192, 384, 768, 1024]
        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained = pretrained)
            # ResNet50 encoder 各层输出通道数
            feat1_c, feat2_c, feat3_c, feat4_c, feat5_c = 64, 256, 512, 1024, 2048
            # unetUp 的输入通道数参考值
            in_filters  = [192, 512, 1024, 3072]
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        
        # 解码器各级的输出通道数
        out_filters = [64, 128, 256, 512]

        # ---------------------------------------------------------
        # 【新增】初始化 GSAG 模块
        # 我们在 feat4, feat3, feat2 的跳跃连接上添加注意力门控。
        # feat1 通常分辨率太高且语义信息少，计算代价大，故省略。
        # ---------------------------------------------------------
        # GSAG4: 过滤 feat4，使用 feat5 作为指导
        # shallow_in=feat4_c, deep_in=feat5_c, 内部融合通道设为 out_filters[3]
        self.gsag4 = GSAGBlock(shallow_in_c=feat4_c, deep_in_c=feat5_c, out_c=out_filters[3])
        
        # GSAG3: 过滤 feat3，使用 up4 的输出作为指导
        # shallow_in=feat3_c, deep_in=out_filters[3], 内部融合通道设为 out_filters[2]
        self.gsag3 = GSAGBlock(shallow_in_c=feat3_c, deep_in_c=out_filters[3], out_c=out_filters[2])

        # GSAG2: 过滤 feat2，使用 up3 的输出作为指导
        # shallow_in=feat2_c, deep_in=out_filters[2], 内部融合通道设为 out_filters[1]
        self.gsag2 = GSAGBlock(shallow_in_c=feat2_c, deep_in_c=out_filters[2], out_c=out_filters[1])
        # ---------------------------------------------------------

        # upsampling 部分 (unetUp 的定义不需要改变，因为 GSAG 不改变浅层特征的通道数)
        # 64,64,512
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])

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
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
        elif self.backbone == "resnet50":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)

        # ---------------------------------------------------
        # 解码器路径 (应用 GSAG)
        # 在拼接前，使用深层特征对浅层特征进行门控过滤
        # ---------------------------------------------------
        
        # Step 4:
        # 使用 feat5 指导过滤 feat4
        feat4_gated = self.gsag4(x_shallow=feat4, g_deep=feat5)
        up4 = self.up_concat4(feat4_gated, feat5)

        # Step 3:
        # 使用 up4 的输出指导过滤 feat3
        feat3_gated = self.gsag3(x_shallow=feat3, g_deep=up4)
        up3 = self.up_concat3(feat3_gated, up4)

        # Step 2:
        # 使用 up3 的输出指导过滤 feat2
        feat2_gated = self.gsag2(x_shallow=feat2, g_deep=up3)
        up2 = self.up_concat2(feat2_gated, up3)

        # Step 1:
        # feat1 通常不加注意力，直接拼接
        up1 = self.up_concat1(feat1, up2)

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