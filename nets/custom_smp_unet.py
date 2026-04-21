import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


#   (FReLU, SELayer)

class FReLU(nn.Module):
    def __init__(self, c1, k=3):
        super().__init__()
        self.conv = nn.Conv2d(c1, c1, k, 1, 1, groups=c1, bias=False)
        self.bn = nn.BatchNorm2d(c1)

    def forward(self, x):
        return torch.max(x, self.bn(self.conv(x)))

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


#  (DEESBlock & GSAGBlock)

class DEESBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DEESBlock, self).__init__()
        mid_channels = out_channels // 4 
        
        self.strip_h = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=(1, 9), stride=1, padding=(0, 4), bias=False),
            nn.BatchNorm2d(mid_channels),
            FReLU(mid_channels)
        )
        self.strip_v = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=(9, 1), stride=1, padding=(4, 0), bias=False),
            nn.BatchNorm2d(mid_channels),
            FReLU(mid_channels)
        )
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False), 
            nn.BatchNorm2d(mid_channels),
            FReLU(mid_channels)
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            FReLU(mid_channels)
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(mid_channels * 4, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            FReLU(out_channels) 
        )

    def forward(self, x):
        b, c, h, w = x.size()
        feat_h = self.strip_h(x)
        feat_v = self.strip_v(x)
        local_avg = self.avg_pool(x)
        diff_feat = x - local_avg 
        feat_edge = self.edge_conv(diff_feat)
        global_feat = self.global_avg_pool(x)
        global_feat = self.global_conv(global_feat)
        global_feat = F.interpolate(global_feat, size=(h, w), mode='nearest')
        concat_feat = torch.cat([feat_h, feat_v, feat_edge, global_feat], dim=1)
        out = self.fusion_conv(concat_feat)
        if x.shape[1] == out.shape[1]:
            out = out + x
        return out

class GSAGBlock(nn.Module):
    def __init__(self, shallow_in_c, deep_in_c, out_c):
        super(GSAGBlock, self).__init__()
        self.shallow_conv = nn.Sequential(
            nn.Conv2d(shallow_in_c, out_c, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.deep_conv = nn.Sequential(
            nn.Conv2d(deep_in_c, out_c, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.se_layer = SELayer(out_c, reduction=8)
        self.psi = nn.Sequential(
            nn.Conv2d(out_c, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.GELU()

    def forward(self, x_shallow, g_deep):
        x_s_aligned = self.shallow_conv(x_shallow)
        g_d_aligned = self.deep_conv(g_deep)
        g_d_upsampled = F.interpolate(g_d_aligned, size=x_s_aligned.size()[2:], mode='bilinear', align_corners=True)
        fusion = x_s_aligned + g_d_upsampled
        fusion = self.relu(fusion)
        fusion_enhanced = self.se_layer(fusion)
        attention_map = self.psi(fusion_enhanced)
        return x_shallow * attention_map



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




class AblationSMPUnet(nn.Module):
    def __init__(self, encoder_name='resnet50', encoder_weights='imagenet', num_classes=2, 
                 use_dees=True, use_gsag=True):
        super(AblationSMPUnet, self).__init__()
        
        self.use_dees = use_dees
        self.use_gsag = use_gsag
        
        # 1. 
        self.encoder = smp.encoders.get_encoder(
            encoder_name, 
            in_channels=3, 
            depth=5, 
            weights=encoder_weights
        )
        encoder_channels = self.encoder.out_channels
        
        out_filters = [64, 128, 256, 512]
        



        if self.use_dees:
         
            self.bottleneck_dees = DEESBlock(encoder_channels[-1], out_filters[3])
        else:
            
            self.bottleneck = nn.Sequential(
                nn.Conv2d(encoder_channels[-1], out_filters[3], kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_filters[3]),
                nn.ReLU(inplace=True)
            )
            
        center_c = out_filters[3]

       
        if self.use_gsag:
            self.gsag4 = GSAGBlock(encoder_channels[-2], center_c, out_filters[3])
            self.gsag3 = GSAGBlock(encoder_channels[-3], out_filters[3], out_filters[2])
            self.gsag2 = GSAGBlock(encoder_channels[-4], out_filters[2], out_filters[1])
        
        self.up_concat4 = unetUp(encoder_channels[-2] + center_c, out_filters[3])
        self.up_concat3 = unetUp(encoder_channels[-3] + out_filters[3], out_filters[2])
        self.up_concat2 = unetUp(encoder_channels[-4] + out_filters[2], out_filters[1])
        self.up_concat1 = unetUp(encoder_channels[-5] + out_filters[1], out_filters[0])

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

    def forward(self, x):
        features = self.encoder(x)
        feat1, feat2, feat3, feat4, feat5 = features[1], features[2], features[3], features[4], features[5]

      
        if self.use_dees:
            #  bottleneck_dees
            feat5_processed = self.bottleneck_dees(feat5)
        else:
            #  bottleneck
            feat5_processed = self.bottleneck(feat5)

        #  Layer 4
        if self.use_gsag:
            feat4_in = self.gsag4(feat4, feat5_processed)
        else:
            feat4_in = feat4
        up4 = self.up_concat4(feat4_in, feat5_processed)

        #  Layer 3
        if self.use_gsag:
            feat3_in = self.gsag3(feat3, up4)
        else:
            feat3_in = feat3
        up3 = self.up_concat3(feat3_in, up4)

        #  Layer 2
        if self.use_gsag:
            feat2_in = self.gsag2(feat2, up3)
        else:
            feat2_in = feat2
        up2 = self.up_concat2(feat2_in, up3)

        up1 = self.up_concat1(feat1, up2)

        final = self.final(up1)
        final = F.interpolate(final, size=x.shape[2:], mode='bilinear', align_corners=True)
        
        return final
