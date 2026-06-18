import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from thop import profile
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# device = 'cpu'

class Conv1DSegmenter(nn.Module):
    def __init__(self, in_channels = 1, num_classes = 1):
        super(Conv1DSegmenter, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, num_classes, kernel_size=1)
        )
    
    def forward(self, x):
        # print(x.shape)
        batch_size, _, H, W = x.shape
        # if x.dim() == 2:
        x = x.unsqueeze(0)  # (1, H, W)
        x = x.view(x.size(0), -1)  # (C, H*W)

        x = self.encoder(x)
        x = self.decoder(x)

        # print(x.shape)
        num_classes ,seq_length = x.shape
      
        x = x.view(batch_size, num_classes, H, W)


        return x  # shape: (batch_size, num_classes, seq_length)




# Bottleneck Block
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False, dilated=1, asymmetric=False, dropout_prob=0.1):
        super().__init__()
        internal_channels = in_channels // 4
        self.downsample = downsample

        self.conv1 = nn.Conv2d(in_channels, internal_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(internal_channels)
        self.prelu1 = nn.PReLU()

        if downsample:
            self.conv2 = nn.Conv2d(internal_channels, internal_channels, kernel_size=3, stride=2, padding=1, bias=False)
        elif asymmetric:
            self.conv2 = nn.Sequential(
                nn.Conv2d(internal_channels, internal_channels, kernel_size=(5, 1), padding=(2, 0), bias=False),
                nn.Conv2d(internal_channels, internal_channels, kernel_size=(1, 5), padding=(0, 2), bias=False),
            )
        elif dilated > 1:
            self.conv2 = nn.Conv2d(internal_channels, internal_channels, kernel_size=3, padding=dilated,
                                   dilation=dilated, bias=False)
        else:
            self.conv2 = nn.Conv2d(internal_channels, internal_channels, kernel_size=3, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(internal_channels)
        self.prelu2 = nn.PReLU()

        self.conv3 = nn.Conv2d(internal_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p=dropout_prob)

        # Optional projection for residual
        self.match_dims = (in_channels != out_channels or downsample)
        if self.match_dims:
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2 if downsample else 1, bias=False)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.prelu2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.dropout(out)

        if self.match_dims:
            residual = self.proj(residual)

        out += residual
        return F.relu(out)










class _ConvBNReLU(nn.Module):
    """Conv-BN-ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class _DSConv(nn.Module):
    """Depthwise Separable Convolutions"""

    def __init__(self, dw_channels, out_channels, stride=1, padding = None, **kwargs):
        super(_DSConv, self).__init__()
        if padding:
                self.conv = nn.Sequential(
                nn.Conv2d(dw_channels, dw_channels, 3, stride, padding=padding, groups=dw_channels, bias=False),
                nn.GroupNorm(num_channels=dw_channels,num_groups=1),
                nn.ReLU(True),
                nn.Conv2d(dw_channels, out_channels, 1, bias=False),
                nn.GroupNorm(num_channels=dw_channels,num_groups=1),
                nn.ReLU(True)
            )
        else:   
            self.conv = nn.Sequential(
                nn.Conv2d(dw_channels, dw_channels, 3, stride, 1, groups=dw_channels, bias=False),
                nn.GroupNorm(num_channels=dw_channels,num_groups=1),
                nn.ReLU(True),
                nn.Conv2d(dw_channels, out_channels, 1, bias=False),
                nn.GroupNorm(num_channels=dw_channels,num_groups=1),
                nn.ReLU(True)
            )

    def forward(self, x):
        return self.conv(x)





class LearningToDownsample(nn.Module):
    """Learning to downsample module"""

    def __init__(self,in_chanels = 3, dw_channels1=32, dw_channels2=48, dw_channels3=64, out_channels=128, **kwargs):
        super(LearningToDownsample, self).__init__()
        # self.conv = _ConvBNReLU(in_chanels, dw_channels1, 3, 2)
        # self.dsconv1 = _DSConv(dw_channels1, dw_channels2, 2)
        # self.dsconv2 = _DSConv(dw_channels2, dw_channels3, 2)
        # self.dsconv3 = _DSConv(dw_channels3, out_channels, 2)
                   
        self.conv = _ConvBNReLU(in_chanels, dw_channels1, 3, 2)
        if dw_channels1!=dw_channels2:
            self.dsconv1 = Bottleneck(dw_channels1, dw_channels2, downsample=True)
        else:
            self.dsconv1 = Bottleneck(dw_channels1, dw_channels2)

        if dw_channels2!=dw_channels3:
            self.dsconv2 = Bottleneck(dw_channels2, dw_channels3, downsample=True)
        else:
            self.dsconv2 = Bottleneck(dw_channels2, dw_channels3)


        if dw_channels3!=out_channels:
             self.dsconv3 = Bottleneck(dw_channels3, out_channels, downsample=True)

        else:
             self.dsconv3 = Bottleneck(dw_channels3, out_channels)




    def forward(self, x):
        x = self.conv(x)
        x = self.dsconv1(x)
        x_2 = self.dsconv2(x)
        x = self.dsconv3(x_2)
        return x, x_2
    


class FeatureFusionModule(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # in_channels = sum of x1.size(1) + x2.size(2)
        self.in_channels = in_channels

        # self.conv_block = nn.Sequential(
        #     nn.Conv2d(in_channels, num_classes, kernel_size=3, stride=1, padding=1), # same padding
        #     nn.BatchNorm2d(num_classes),
        #     nn.ReLU()
        # )
        self.conv_block = _DSConv(in_channels, num_classes, kernel_size=3, stride=1, padding=1)

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.conv_1x1 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def forward(self, x1, x2):
        if x1.shape[2:] != x2.shape[2:]:
            x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x1, x2], dim=1)
        assert x.size(1) == self.in_channels, "in_channels and concatenated feature channel must be the same"

        # feature = self.conv_block(x)

        # attn = self.avg_pool(feature)
        # attn = self.conv_1x1(attn)
        # attn = self.relu(attn)
        # attn = self.conv_1x1(attn)
        # attn = self.relu(attn)

        # mul = torch.mul(feature, attn)
        # x = torch.add(feature, mul)

        return x








class LowLevelFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.bottleneck1 = nn.Sequential(
            *[Bottleneck(in_channels, out_channels) for _ in range(4)],
        )

        self.bottleneck2 = nn.Sequential(
            *[
                Bottleneck(out_channels, out_channels),
                Bottleneck(out_channels, out_channels, dilated=2),
                Bottleneck(out_channels, out_channels, asymmetric=True),
                Bottleneck(out_channels, out_channels, dilated=4),
                Bottleneck(out_channels, out_channels),
                Bottleneck(out_channels, out_channels, dilated=8),
                Bottleneck(out_channels, out_channels, asymmetric=True),
                Bottleneck(out_channels, out_channels, dilated=16),
            ]
        )

      

    def forward(self, x):
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)

        return x



class Reduce_DSC(nn.Module):
    def __init__(self,in_chanels, out_chanels):
        super().__init__()
        self.block = _DSConv(in_chanels, out_chanels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.block(x)
    


class FastAttention(nn.Module):
    def __init__(self, in_channels: int, embed_channels: int):
        """
        in_channels: liczba kanałów wejściowych c
        embed_channels: liczba kanałów dla Q i K (c')
        """
        super().__init__()
        self.to_q = nn.Conv2d(in_channels, embed_channels, kernel_size=1, bias=False)
        self.to_k = nn.Conv2d(in_channels, embed_channels, kernel_size=1, bias=False)
        self.to_v = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)

    def forward(self, x):
        # x: Tensor o wymiarach (B, C, H, W)
        B, C, H, W = x.shape
        n = H * W

        # Oblicz Q, K, V
        Q = self.to_q(x).view(B, -1, n)       # (B, c', n)
        K = self.to_k(x).view(B, -1, n)       # (B, c', n)
        V = self.to_v(x).view(B, C, n)        # (B, C, n)

        # L2-normalizacja wzdłuż kanałów
        Q_norm = F.normalize(Q, dim=1)        # (B, c', n)
        K_norm = F.normalize(K, dim=1)        # (B, c', n)

        # Fast Attention: Y = (1/n) * Q_norm @ (K_norm^T @ V)
        # 1) najpierw (c'×n) @ (n×C) → (c'×C) pozycje
        intermediate = torch.einsum('bcn,bun->bcu', K_norm, V)  # (B, c', C)

        # 2) następnie (B, c', n) @ (B, c', C) → (B, n, C)
        Y = torch.einsum('bcn,bcu->bnu', Q_norm, intermediate)   # (B, n, C)

        # Średni dzielnik 1/n
        Y = Y / n

        # Zmień kształt z powrotem na (B, C, H, W)
        Y = Y.permute(0, 2, 1).view(B, C, H, W)
        return Y









class FAscnn_pp_V1(nn.Module):
    def __init__(self,in_chanels, num_classes):
        super(model.__class__.__name__1, self).__init__()
        self.learning_to_downsample = LearningToDownsample(in_chanels,16, 32, 64, 128)
        self.MLE1 = LowLevelFeatureExtractor(64,64)

        self.LLE1 = LowLevelFeatureExtractor(128,128)
        self.reduce1 = Reduce_DSC(128,64)
        self.FFM = FeatureFusionModule(128,num_classes)
        # self.global_feature_extractor = GlobalFeatureExtractor(128, [128, 160, 184], 64, 6, [3, 3, 3])

        self.up = nn.Sequential(
            Bottleneck(128, 64),
            *[
                Bottleneck(64, 64, dilated=1),
                Bottleneck(64, 64, dilated=1),
                Bottleneck(64, 16),
                Bottleneck(16, 16,  dilated=1)
            ]
        )


        # self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        # self.deconv2 = nn.ConvTranspose2d(64, 16, kernel_size=2, stride=2)
        # self.classifier = nn.ConvTranspose2d(16, num_classes, kernel_size=2, stride=2)

        self.classifier = nn.ConvTranspose2d(in_channels=16, out_channels=num_classes, kernel_size=3, 
                                           stride=2, padding=1, output_padding=1, bias=False)
            

    def forward(self, x):
        x_input = x
        # size = x.size()[2:]
        lower_res_features, midle_res_features = self.learning_to_downsample(x)
        midle_res_features = self.MLE1(midle_res_features)
        # x = self.global_feature_extractor(lower_res_features)
        x = self.LLE1(lower_res_features)
        x = self.reduce1(x)


        x = self.FFM(x, midle_res_features)
        x = self.up(x)
        # x = self.deconv1(x)
        # x = self.deconv2(x)
        x = self.classifier(x)
        

        x = F.interpolate(x, size=x_input.shape[2:], mode='bilinear', align_corners=True)

        return x



class FAscnn_pp_V2(nn.Module):
    def __init__(self,in_chanels, num_classes):
        super(FAscnn_pp_V2, self).__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(in_chanels, 16, kernel_size=3, stride=2, padding=1),
            *[
                Bottleneck(16, 16,  dilated=1),
                Bottleneck(16, 32, downsample=True),
                Bottleneck(32, 32)
            ]
        )
        self.FA1 = FastAttention(32,16)


        self.down2 = nn.Sequential(
            Bottleneck(32, 64, downsample=True),
            *[
                Bottleneck(64, 64),
                Bottleneck(64, 64, dilated=2),
                Bottleneck(64, 64, asymmetric=True),
                Bottleneck(64, 64, dilated=4),
                Bottleneck(64, 64),
                Bottleneck(64, 64, dilated=8),
                Bottleneck(64, 64, asymmetric=True),
                Bottleneck(64, 64, dilated=16),
            ]
        )
        self.down3 = nn.Sequential(
            Bottleneck(64, 128, downsample=True),
            *[
                Bottleneck(128, 128),
                Bottleneck(128, 128, dilated=2),
                Bottleneck(128, 128, asymmetric=True),
                Bottleneck(128, 128, dilated=4),
                Bottleneck(128, 128),
                Bottleneck(128, 128, dilated=8),
                Bottleneck(128, 128, asymmetric=True),
                Bottleneck(128, 128, dilated=16),
            ]
        )
        self.FA2 = FastAttention(128, 32)


        self.transpose1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, 
                                           stride=2, padding=1, output_padding=1, bias=False)
            
        self.transpose2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, 
                                           stride=2, padding=1, output_padding=1, bias=False)
        



            
        self.classifier1 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, 
                                           stride=2, padding=1, output_padding=1, bias=False)
        self.classifier2 = nn.ConvTranspose2d(in_channels=32, out_channels=num_classes, kernel_size=3, 
                                           stride=2, padding=1, output_padding=1, bias=False)

    def forward(self, x):
        x = self.down1(x)
        x_residual =  x
        x_residual = self.FA1(x_residual)


        x = self.down2(x)
        x = self.down3(x)
        x = self.FA2(x)
        x = self.transpose1(x)
        x = self.transpose2(x)

        x = torch.cat([x, x_residual], dim=1)
        x = self.classifier1(x)
        x = self.classifier2(x)
        # x = torch.sigmoid(x)       


        return x





class FAscnn_pp_V3(nn.Module):
    def __init__(self,in_channels, num_classes):
        super(FAscnn_pp_V3, self).__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            *[
                Bottleneck(16, 16,  dilated=1),
                Bottleneck(16, 32, downsample=True),
                Bottleneck(32, 32)
            ]
        )
        self.FA1 = FastAttention(32,8)


        self.down2 = nn.Sequential(
            Bottleneck(32, 64, downsample=True),
            *[
                Bottleneck(64, 64),
                Bottleneck(64, 64, dilated=2),
                Bottleneck(64, 64, asymmetric=True),
                Bottleneck(64, 64, dilated=4),
                Bottleneck(64, 64),
                Bottleneck(64, 64, dilated=8),
                Bottleneck(64, 64, asymmetric=True),
                Bottleneck(64, 64, dilated=16),
            ]
        )

        self.FA2 = FastAttention(64, 16)


        self.down3 = nn.Sequential(
            Bottleneck(64, 128, downsample=True),
            *[
                Bottleneck(128, 128),
                Bottleneck(128, 128, dilated=2),
                Bottleneck(128, 128, asymmetric=True),
                Bottleneck(128, 128, dilated=4),
                Bottleneck(128, 128),
                Bottleneck(128, 128, dilated=8),
                Bottleneck(128, 128, asymmetric=True),
                Bottleneck(128, 128, dilated=16),
            ]
        )

        self.FA3 = FastAttention(128, 32)


        self.transpose1_1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, 
                                           stride=2, padding=1, output_padding=1, bias=False)
            
            
        self.transpose1_2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, 
                                           stride=2, padding=1, output_padding=1, bias=False)
        



            
        self.classifier1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, 
                                           stride=2, padding=1, output_padding=1, bias=False)
        self.classifier2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, 
                                           stride=2, padding=1, output_padding=1, bias=False)
        self.final = nn.ConvTranspose2d(in_channels=32, out_channels=num_classes, kernel_size=3, 
                                           stride=2, padding=1, output_padding=1, bias=False)

    def forward(self, x):
        x = self.down1(x)
        x_residual =  x
        x_residual = self.FA1(x_residual)


        x = self.down2(x)
        x_residual2 = x 
        x = self.down3(x)
        x = self.FA3(x)

        x_residual2 = self.FA2(x_residual2)


        x = self.transpose1_1(x)
        x = self.transpose1_2(x)

        x = torch.cat([x, x_residual], dim=1)
        x_residual2 = F.interpolate(x_residual2, size=x.shape[2:])
        x = torch.cat([x, x_residual2], dim=1)
        x = self.classifier1(x)
        x = self.classifier2(x)
        x = self.final(x)
        x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)


        return x
    

    

class _ConvLNReLU(nn.Module):
    """Conv-BN-ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, **kwargs):
        super(_ConvLNReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.GroupNorm(num_groups=1, num_channels= out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)

class _DSConvV2(nn.Module):
    """Depthwise Separable Convolutions"""

    def __init__(self, dw_channels, out_channels, stride=1, padding = None,asymmetric=False):
        super(_DSConvV2, self).__init__()
        if asymmetric:
                self.conv2 = nn.Sequential(
                nn.Conv2d(dw_channels, dw_channels, kernel_size=(5, 1), padding=(2, 0), bias=False),
                nn.Conv2d(dw_channels, dw_channels, kernel_size=(1, 5), padding=(0, 2), bias=False),
                nn.GroupNorm(num_channels=dw_channels,num_groups=1),
                nn.ReLU(),
                nn.Conv2d(dw_channels, out_channels, 1, bias=False),
                nn.GroupNorm(num_channels=dw_channels,num_groups=1),
                nn.ReLU()
                )

        if padding:
                self.conv = nn.Sequential(

                nn.Conv2d(dw_channels, dw_channels, 3, stride, padding=padding, groups=dw_channels, bias=False),
                nn.GroupNorm(num_channels=dw_channels,num_groups=1),
                nn.ReLU(),
                nn.Conv2d(dw_channels, out_channels, 1, bias=False),
                nn.GroupNorm(num_channels=dw_channels,num_groups=1),
                nn.ReLU()
            )
        else:   
            self.conv = nn.Sequential(
                nn.Conv2d(dw_channels, dw_channels, 3, stride, 1, groups=dw_channels, bias=False),
                nn.GroupNorm(num_channels=dw_channels,num_groups=1),
                nn.ReLU(),
                nn.Conv2d(dw_channels, out_channels, 1, bias=False),
                nn.GroupNorm(num_channels=dw_channels,num_groups=1),
                nn.ReLU()
            )

    def forward(self, x):
        return self.conv(x)


class DSC_block(nn.Module):
    def __init__(self,in_chanels, out_channels, downsample=True, asymmetric= True):
        super(DSC_block, self).__init__()
        self.downsample = downsample
        self.conv1 = _ConvLNReLU(in_channels=in_chanels, out_channels=in_chanels,kernel_size=1)
        if downsample:
            self.DSC_1 = _DSConvV2(in_chanels,in_chanels,stride=2,padding=1)
        else:
            if asymmetric:
                 self.DSC_1 = _DSConvV2(in_chanels,in_chanels,asymmetric=True)
                
            else:
                self.DSC_1 = _DSConvV2(in_chanels,out_channels,stride=1,padding=1)

        self.relu = nn.ReLU()
        self.conv2 = _ConvLNReLU(in_channels=in_chanels, out_channels=out_channels,kernel_size=1)
    def forward(self, x):
        x_residual = []
        x = self.conv1(x)
        x_residual.append(x)
        x = self.DSC_1(x)
        x = self.relu(x)
        x_residual.append(x)
        x = self.conv2(x)
        x_residual.append(x)


        if not self.downsample: 
            for i in x_residual:
                x= x +i

        return x 

class FAscnn_pp_V4(nn.Module):
    def __init__(self,in_chanels, num_classes):
        super(FAscnn_pp_V4, self).__init__()
        self.conv1 =  nn.Conv2d(in_chanels, 16, kernel_size=3, stride=2, padding=1)

        self.block_1 = DSC_block(16,32,True)
        self.block_1_5 = nn.Sequential(
            DSC_block(32,32,False),
            DSC_block(32,32,False),
            DSC_block(32,32,False)


        ) 
        self.FA1 = FastAttention(32,16)
        self.block_2 = DSC_block(32,64,True)
        self.block_2_5 = nn.Sequential(
            DSC_block(64,64,False),
            DSC_block(64,64,False),
            DSC_block(64,64,False)


        )
        self.block_3 = DSC_block(64,128,True)
        self.block_3_5 = nn.Sequential(
            DSC_block(128,128,False),
            DSC_block(128,128,False),
            DSC_block(128,128,False)


        )
       
        self.transpose1_1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, 
                                           stride=2, padding=1, output_padding=1, bias=False)


        self.FA2 = FastAttention(32, 16)
        self.block_4 = DSC_block(32,64,True)
        self.block_4_5 = nn.Sequential(
            DSC_block(64,64,False),
            DSC_block(64,64,False),
            DSC_block(64,64,False)


        )
        # self.block_5 = DSC_block(64,128,True)


        self.FA3 = FastAttention(16, 8)
        self.block_5 = DSC_block(16,32,True)
        self.block_5_5 = nn.Sequential(
            DSC_block(32,32,False),
            DSC_block(32,32,False),
            DSC_block(32,32,False)


        )
        self.block_6 = DSC_block(32,64,True)
        self.block_6_5 = nn.Sequential(
            DSC_block(64,64,False),
            DSC_block(64,64,False),
            DSC_block(64,64,False)


        )
       

      



            

        self.final1 = nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=3, 
                                           stride=2, padding=1, output_padding=1, bias=False)
        self.final2 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, 
                                           stride=2, padding=1, output_padding=1, bias=False)
        self.final3 = nn.ConvTranspose2d(in_channels=16, out_channels=num_classes, kernel_size=3, 
                                           stride=2, padding=1, output_padding=1, bias=False)









    def forward(self, x):
        x = self.conv1(x)
        x_res = x
        """First branch"""
        x = self.block_1(x)
        x = self.block_1_5(x)
        x_res2 = x
        x = self.FA1(x)
        x = self.block_2(x)
        x = self.block_2_5(x)
        x = self.block_3(x)
        x = self.block_3_5(x)
        x = self.transpose1_1(x)

        # print(f"1 branch x.shape : {x.shape}")

        """Second branch"""
        x_res2 = self.FA2(x_res2)
        x_res2 = self.block_4(x_res2)
        x = self.block_4_5(x_res2)
        # x_res2 = self.block_5(x_res2)
        # x = x + x_res2
        x = torch.cat([x, x_res2], dim=1)


        # print(f"2 branch x.shape : {x_res2.shape}")
        # print(f"1 branch x.shape : {x.shape}")
    

        """Third Branch"""
        x_res = self.FA3(x_res)
        x_res = self.block_5(x_res)
        x = self.block_5_5(x_res)
        x_res = self.block_6(x_res)
        x = self.block_6_5(x_res)

        x = x + x_res

        x = torch.cat([x, x_res], dim=1)


        # print(f"3 branch x.shape : {x_res.shape}")
        # print(f"1 branch x.shape : {x.shape}")

        x = self.final1(x)
        x = self.final2(x)
        x = self.final3(x)

        return x





class FAscnn_pp_V5(nn.Module):
    def __init__(self,in_chanels, num_classes):
        super(FAscnn_pp_V5, self).__init__()

        self.block_1= nn.Sequential(
            DSC_block(in_chanels,16,True),
            DSC_block(16,32,True),
            DSC_block(32,64,True),
            DSC_block(64,128,True)


        ) 
        self.FFM = FeatureFusionModule(in_channels=64 * 2,num_classes=num_classes)

        self.block_2= nn.Sequential(
            DSC_block(in_chanels,16,True),
            DSC_block(16,32,True),
            DSC_block(32,64,True)
        ) 
        self.block_3=  DSC_block(64,128,True)
        self.FA1 = FastAttention(64,16)
        self.FA2 = FastAttention(128,32)
            

        self.transpose1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, 
                                           stride=2, padding=1, output_padding=1, bias=False)
        self.transpose2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, 
                                           stride=2, padding=1, output_padding=1, bias=False)
        





        # self.final = nn.ConvTranspose2d(in_channels=128, out_channels=num_classes, kernel_size=3, 
        #                                    stride=2, padding=2, output_padding=1, bias=False)

        self.final = nn.Sequential( 
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, 
                                           stride=2, padding=1, output_padding=1, bias=False),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, 
                                           stride=2, padding=1, output_padding=1, bias=False),
            nn.ConvTranspose2d(in_channels=32, out_channels=num_classes, kernel_size=3, 
                                           stride=2, padding=1, output_padding=1, bias=False)                  
                                            
                                            
            )      








    def forward(self, x):
        x_res = x
        """First branch"""
        x = self.block_1(x)
        x = self.transpose1(x)


        """Second branch"""
        x_res1 = self.block_2(x_res)
        x_res2 = self.block_3(x_res1)
        x_res1 = self.FA1(x_res1)
        x_res2 = self.FA2(x_res2)
        x_res2 = self.transpose2(x_res2)
        x_res2 = F.interpolate(x_res2, size=x_res1.shape[2:], mode='bilinear', align_corners=False)


        x_res = torch.cat([x_res1,x_res2],dim=1)
        
        x_res = x_res1+x_res2


        x = F.interpolate(x, size=x_res.shape[2:], mode='bilinear', align_corners=False)

    
        x = self.FFM(x,x_res)

        x = self.final(x)
        


        return x


class UBottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu', down_ratio=4):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.down_channels = int(self.in_channels / down_ratio)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'mish':
            self.activation = nn.Mish()

        self.main_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1)

        self.convt1 = nn.ConvTranspose2d(in_channels=self.in_channels, out_channels=self.down_channels, kernel_size=1,
                                         padding=0, bias=False)
        self.activation1 = self.activation
        self.convt2 = nn.ConvTranspose2d(in_channels=self.down_channels, out_channels=self.down_channels, kernel_size=3,
                                         stride=2, padding=1, 
                                         output_padding=1, bias=False)
        self.activation2 = self.activation
        self.convt3 = nn.ConvTranspose2d(in_channels=self.down_channels, out_channels=self.out_channels, kernel_size=1,
                                         padding=0, bias=False)
        self.activation3 = self.activation

        self.norm1 = nn.BatchNorm2d(self.down_channels)
        self.norm2 = nn.BatchNorm2d(self.down_channels)
        self.norm3 = nn.BatchNorm2d(self.out_channels)

        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            # nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1),
        )
        self.drop_out = nn.Dropout2d(p=0.1)
        

    def forward(self, x, indices):
        x_main = x

        # Side branch
        x = self.convt1(x)
        x = self.norm1(x)
        x = self.activation1(x)

        x = self.convt2(x)
        x = self.norm2(x)
        x = self.activation2(x)

        x = self.convt3(x)
        x = self.norm3(x)

        x = self.drop_out(x)

        # Main branch
        x_main = self.main_conv(x_main)
        # x_main = self.unpool(x_main, indices, output_size=x.size())
        x_main = self.up(x_main)

        # Concatenate
        x = x + x_main
        x = self.activation3(x)

        return x

class DDBottleNeck(nn.Module):
    """
    Bottleneck 2.x for dowmsampling and dilated types
    """

    def __init__(self, in_channels, out_channels, dilation, downsampling, activation='relu', down_ratio=4, p=0.1):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilation = dilation
        self.downsampling = downsampling

        if self.downsampling:  # if bottleneck 2.x with downsampling
            self.stride = 2
            self.down_channels = int(in_channels // down_ratio)
        else:
            self.stride = 1
            self.down_channels = int(out_channels // down_ratio)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'mish':
            self.activation = nn.Mish()

        # Main MaxPooling
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, return_indices=True)

        # Side Conv 1x1
        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.down_channels, kernel_size=1, stride=1, \
                               padding=0, bias=False, dilation=1)
        self.activation1 = self.activation
        self.conv2 = nn.Conv2d(in_channels=self.down_channels, out_channels=self.down_channels, kernel_size=3,
                               stride=self.stride, \
                               padding=self.dilation, bias=True, dilation=self.dilation)
        self.activation2 = self.activation
        self.conv3 = nn.Conv2d(in_channels=self.down_channels, out_channels=self.out_channels, kernel_size=1, stride=1, \
                               padding=0, bias=False, dilation=1)
        self.activation3 = self.activation
        self.norm1 = nn.BatchNorm2d(self.down_channels)
        self.norm2 = nn.BatchNorm2d(self.down_channels)
        self.norm3 = nn.BatchNorm2d(self.out_channels)

        self.dropout = nn.Dropout2d(p=p)

    def forward(self, x):
        batch_size = x.size()[0]
        x_main = x

        # Side branch
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation2(x)

        x = self.conv3(x)
        x = self.norm3(x)

        x = self.dropout(x)

        # Main branch
        if self.downsampling:
            x_main, indices = self.max_pool(x_main)

        if self.in_channels != self.out_channels:
            out_shape = self.out_channels - self.in_channels
            temp = torch.zeros((batch_size, out_shape, x.shape[2], x.shape[3]))
            if x_main.device.type != 'cpu':
                temp = temp.cuda()
            x_main = torch.cat((x_main, temp), dim=1)

        # Side + Main
        x = x + x_main
        x = self.activation3(x)

        if self.downsampling:
            return x, indices
        else:
            return x

class ABottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu', down_ratio=4):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.down_channels = int(self.in_channels / down_ratio)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'mish':
            self.activation = nn.Mish()

        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.down_channels, kernel_size=1, stride=1, \
                               padding=0, bias=False)
        self.activation1 = self.activation

        self.conv21 = nn.Conv2d(in_channels=self.down_channels, out_channels=self.down_channels, kernel_size=(1, 5),
                                stride=1, \
                                padding=(0, 2), bias=False)
        self.conv22 = nn.Conv2d(in_channels=self.down_channels, out_channels=self.down_channels, kernel_size=(5, 1),
                                stride=1, \
                                padding=(2, 0), bias=False)
        self.activation2 = self.activation

        self.conv3 = nn.Conv2d(in_channels=self.down_channels, out_channels=self.out_channels, kernel_size=1, stride=1, \
                               padding=0, bias=False)
        self.activation3 = self.activation

        self.norm1 = nn.BatchNorm2d(self.down_channels)
        self.norm2 = nn.BatchNorm2d(self.down_channels)
        self.norm3 = nn.BatchNorm2d(self.out_channels)

        self.drop_out = nn.Dropout2d(p=0.1)

    def forward(self, x):
        batch_size = x.size()[0]
        x_main = x

        # Side branch
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation1(x)

        x = self.conv21(x)
        x = self.conv22(x)
        x = self.norm2(x)
        x = self.activation2(x)

        x = self.conv3(x)

        x = self.drop_out(x)
        x = self.norm3(x)

        # Main branch
        if self.in_channels != self.out_channels:
            out_shape = self.out_channels - self.in_channels
            temp = torch.zeros((batch_size, out_shape, x.shape[2], x.shape[3]))
            if torch.cuda.is_available():
                temp = temp.cuda()
            x_main = torch.cat((x_main, temp), dim=1)

        # Side + Main
        x += x_main
        x = self.activation3(x)
        return x



class FeatureFusionModulev6(nn.Module):
    def __init__(self, spatial_channels, context_channels, out_channels):
        super().__init__()
        self.convblk = nn.Sequential(
            nn.Conv2d(spatial_channels + context_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Prosty attention (opcjonalnie) — można pominąć, by zachować lekkość
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 4, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, spatial_feat, context_feat):
        # Dopasuj rozmiar cech kontekstowych do przestrzennych
        if context_feat.size()[2:] != spatial_feat.size()[2:]:
            context_feat = F.interpolate(context_feat, size=spatial_feat.size()[2:], mode='bilinear', align_corners=False)
        
        feat = torch.cat([spatial_feat, context_feat], dim=1)
        feat = self.convblk(feat)
        
        # Lekka uwaga na wagę cech (możesz to pominąć, jeśli chcesz maksymalnej szybkości)
        attn = self.attention(feat)
        feat = feat * attn
        
        return feat


class FAscnn_pp_V6(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
        use_fa1=True,
        use_fa2=True,
        use_fa3=True,
        use_second_branch=True,
    ):
        super(FAscnn_pp_V6, self).__init__()
        self.use_second_branch = use_second_branch
        self.use_fa1 = use_fa1
        self.use_fa2 = use_fa2 and use_second_branch
        self.use_fa3 = use_fa3 and use_second_branch


        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # self.conv_1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)

        # self.b10 = DDBottleNeck(in_channels=16, out_channels=64, dilation=1, downsampling=True, p=0.01)
        
        self.b11 = DDBottleNeck(in_channels=64, out_channels=64, dilation=1, downsampling=False, p=0.01)
        self.b12 = DDBottleNeck(in_channels=64, out_channels=64, dilation=1, downsampling=False, p=0.01)
        self.b13 = DDBottleNeck(in_channels=64, out_channels=64, dilation=1, downsampling=False, p=0.01)
        self.b14 = DDBottleNeck(in_channels=64, out_channels=64, dilation=1, downsampling=False, p=0.01)

        # The second bottleneck
        self.b20 = DDBottleNeck(in_channels=64, out_channels=128, dilation=1, downsampling=True)
        self.b21 = DDBottleNeck(in_channels=128, out_channels=128, dilation=1, downsampling=False)
        self.b22 = DDBottleNeck(in_channels=128, out_channels=128, dilation=2, downsampling=False)
        self.b23 = ABottleNeck(in_channels=128, out_channels=128)

        self.b24 = DDBottleNeck(in_channels=128, out_channels=128, dilation=4, downsampling=False)
        self.b25 = DDBottleNeck(in_channels=128, out_channels=128, dilation=1, downsampling=False)
        self.b26 = DDBottleNeck(in_channels=128, out_channels=128, dilation=8, downsampling=False)
        self.b27 = ABottleNeck(in_channels=128, out_channels=128)
        self.b28 = DDBottleNeck(in_channels=128, out_channels=128, dilation=16, downsampling=False)

        # The third bottleneck
        self.b31 = DDBottleNeck(in_channels=128, out_channels=128, dilation=1, downsampling=False)
        self.b32 = DDBottleNeck(in_channels=128, out_channels=128, dilation=2, downsampling=False)
        self.b33 = ABottleNeck(in_channels=128, out_channels=128)

        self.b34 = DDBottleNeck(in_channels=128, out_channels=128, dilation=4, downsampling=False)
        self.b35 = DDBottleNeck(in_channels=128, out_channels=128, dilation=1, downsampling=False)
        self.b36 = DDBottleNeck(in_channels=128, out_channels=128, dilation=8, downsampling=False)
        self.b37 = ABottleNeck(in_channels=128, out_channels=128)
        self.b38 = DDBottleNeck(in_channels=128, out_channels=128, dilation=16, downsampling=False)

        # The fourth bottleneck
        self.b40 = UBottleNeck(in_channels=128, out_channels=64)
        self.b41 = DDBottleNeck(in_channels=64, out_channels=64, dilation=1, downsampling=False)
        self.b42 = DDBottleNeck(in_channels=64, out_channels=64, dilation=1, downsampling=False)
        self.b50 = UBottleNeck(in_channels=64, out_channels=16)
        self.b51 = DDBottleNeck(in_channels=16, out_channels=16, dilation=1, downsampling=False)



        self.FA1 = FastAttention(128, 32) if self.use_fa1 else nn.Identity()

        if self.use_second_branch:
            self.conv_2 =  nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1)
            self.FA2 = FastAttention(16, 1) if self.use_fa2 else nn.Identity()

            self.b61 = DDBottleNeck(in_channels=16, out_channels=16, dilation=1, downsampling=False, p=0.01)
            self.b62 = DDBottleNeck(in_channels=16, out_channels=16, dilation=1, downsampling=False, p=0.01)
            self.b63 = DDBottleNeck(in_channels=16, out_channels=64, dilation=1, downsampling=True, p=0.01)
            self.b64 = DDBottleNeck(in_channels=64, out_channels=64, dilation=1, downsampling=False, p=0.01)
            self.b65 = DDBottleNeck(in_channels=64, out_channels=64, dilation=1, downsampling=False, p=0.01)

            self.FA3 = FastAttention(64, 16) if self.use_fa3 else nn.Identity()

            self.b71 = DDBottleNeck(in_channels=64, out_channels=64, dilation=1, downsampling=False)
            self.b72 = DDBottleNeck(in_channels=64, out_channels=64, dilation=1, downsampling=False)
            self.b73 = UBottleNeck(in_channels=64, out_channels=16)
            self.b74 = DDBottleNeck(in_channels=16, out_channels=16, dilation=1, downsampling=False)
        else:
            self.conv_2 = None
            self.FA2 = nn.Identity()
            self.b61 = None
            self.b62 = None
            self.b63 = None
            self.b64 = None
            self.b65 = None
            self.FA3 = nn.Identity()
            self.b71 = None
            self.b72 = None
            self.b73 = None
            self.b74 = None

        final_in_channels = 32 if self.use_second_branch else 16
        self.final = nn.ConvTranspose2d(
            in_channels=final_in_channels,
            out_channels=num_classes,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False,
        )

    def forward(self, x):
        x_res = x
        """"First Branch"""
        x = self.conv_1(x)
        i1 = None
        # x, i1 = self.b10(x) 
        x = self.b11(x)
        x = self.b12(x)
        x = self.b13(x)
        x = self.b14(x)

        # The second bottleneck
        x, i2 = self.b20(x)
        x = self.b21(x)
        x = self.b22(x)
        x = self.b23(x)
        x = self.b24(x)
        x = self.b25(x)
        x = self.b26(x)
        x = self.b27(x)
        x = self.b28(x)

        # The third bottleneck
        x = self.b31(x)
        x = self.b32(x)
        x = self.b33(x)
        x = self.b34(x)
        x = self.b35(x)
        x = self.b36(x)
        x = self.b37(x)
        x = self.b38(x)

        x = self.FA1(x)

        # The fourth bottleneck
        x = self.b40(x, i2)
        x = self.b41(x)
        x = self.b42(x)

        # The fifth bottleneck
        x = self.b50(x, i1)
        x = self.b51(x)


        x = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=False)

        if self.use_second_branch:
            """"Second Branch"""
            x_res = self.conv_2(x_res)
            
            x_res = self.FA2(x_res)

            x_res = self.b61(x_res)
            x_res = self.b62(x_res)
            x_res, i4 = self.b63(x_res)
            x_res = self.b64(x_res)
            x_res = self.b65(x_res)

            x_res = self.FA3(x_res)

            x_res = self.b71(x_res)
            x_res = self.b72(x_res)

            x_res = self.b73(x_res, i4)
            x_res = self.b74(x_res)

            x = torch.cat([x, x_res], dim=1)

        x = self.final(x)

            
        return x
    



class FAscnn_pp_V7(nn.Module):
    def __init__(self,in_chanels, num_classes):
        super(FAscnn_pp_V7, self).__init__()
        self.conv =  nn.Conv2d(in_chanels, 16, kernel_size=3, stride=2, padding=1)


        self.b10 = DDBottleNeck(in_channels=16, out_channels=64, dilation=1, downsampling=True, p=0.01)
        self.b11 = DDBottleNeck(in_channels=64, out_channels=64, dilation=1, downsampling=False, p=0.01)
        self.b12 = DDBottleNeck(in_channels=64, out_channels=64, dilation=1, downsampling=False, p=0.01)
        self.b13 = DDBottleNeck(in_channels=64, out_channels=64, dilation=1, downsampling=False, p=0.01)
        self.b14 = DDBottleNeck(in_channels=64, out_channels=64, dilation=1, downsampling=False, p=0.01)

        # The second bottleneck
        self.b20 = DDBottleNeck(in_channels=64, out_channels=128, dilation=1, downsampling=True)
        self.b21 = DDBottleNeck(in_channels=128, out_channels=128, dilation=1, downsampling=False)
        self.b22 = DDBottleNeck(in_channels=128, out_channels=128, dilation=2, downsampling=False)
        self.b23 = ABottleNeck(in_channels=128, out_channels=128)

        self.b24 = DDBottleNeck(in_channels=128, out_channels=128, dilation=4, downsampling=False)
        self.b25 = DDBottleNeck(in_channels=128, out_channels=128, dilation=1, downsampling=False)
        self.b26 = DDBottleNeck(in_channels=128, out_channels=128, dilation=8, downsampling=False)
        self.b27 = ABottleNeck(in_channels=128, out_channels=128)
        self.b28 = DDBottleNeck(in_channels=128, out_channels=128, dilation=16, downsampling=False)

        # The third bottleneck
        self.b31 = DDBottleNeck(in_channels=128, out_channels=128, dilation=1, downsampling=False)
        self.b32 = DDBottleNeck(in_channels=128, out_channels=128, dilation=2, downsampling=False)
        self.b33 = ABottleNeck(in_channels=128, out_channels=128)

        self.b34 = DDBottleNeck(in_channels=128, out_channels=128, dilation=4, downsampling=False)
        self.b35 = DDBottleNeck(in_channels=128, out_channels=128, dilation=1, downsampling=False)
        self.b36 = DDBottleNeck(in_channels=128, out_channels=128, dilation=8, downsampling=False)
        self.b37 = ABottleNeck(in_channels=128, out_channels=128)
        self.b38 = DDBottleNeck(in_channels=128, out_channels=128, dilation=16, downsampling=False)

        # The fourth bottleneck
        self.b40 = UBottleNeck(in_channels=128, out_channels=64)
        self.b41 = DDBottleNeck(in_channels=64, out_channels=64, dilation=1, downsampling=False)
        self.b42 = DDBottleNeck(in_channels=64, out_channels=64, dilation=1, downsampling=False)
        self.b50 = UBottleNeck(in_channels=64, out_channels=16)
        self.b51 = DDBottleNeck(in_channels=16, out_channels=16, dilation=1, downsampling=False)



        self.FA1 = FastAttention(128, 32)


        self.FA2 = FastAttention(16,8)

        self.b61 = DDBottleNeck(in_channels=16, out_channels=16, dilation=1, downsampling=False, p=0.01)
        self.b62 = DDBottleNeck(in_channels=16, out_channels=16, dilation=1, downsampling=False, p=0.01)
        self.b63 = DDBottleNeck(in_channels=16, out_channels=64, dilation=1, downsampling=True, p=0.01)
        self.b64 = DDBottleNeck(in_channels=64, out_channels=64, dilation=1, downsampling=False, p=0.01)
        self.b65 = DDBottleNeck(in_channels=64, out_channels=64, dilation=1, downsampling=False, p=0.01)

        self.FA3 = FastAttention(64,16)



        self.b71 = DDBottleNeck(in_channels=64, out_channels=64, dilation=1, downsampling=False)
        self.b72 = DDBottleNeck(in_channels=64, out_channels=64, dilation=1, downsampling=False)
        
        self.b73 = UBottleNeck(in_channels=64, out_channels=16)
        self.b74 = DDBottleNeck(in_channels=16, out_channels=16, dilation=1, downsampling=False)


        self.final = nn.ConvTranspose2d(in_channels=32, out_channels=num_classes, kernel_size=3, 
                                           stride=2, padding=1, output_padding=1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        x_res = x
        """"First Branch"""
        x, i1 = self.b10(x)
        x = self.b11(x)
        x = self.b12(x)
        x = self.b13(x)
        x = self.b14(x)

        # The second bottleneck
        x, i2 = self.b20(x)
        x = self.b21(x)
        x = self.b22(x)
        x = self.b23(x)
        x = self.b24(x)
        x = self.b25(x)
        x = self.b26(x)
        x = self.b27(x)
        x = self.b28(x)

        # The third bottleneck
        x = self.b31(x)
        x = self.b32(x)
        x = self.b33(x)
        x = self.b34(x)
        x = self.b35(x)
        x = self.b36(x)
        x = self.b37(x)
        x = self.b38(x)

        x = self.FA1(x)

        # The fourth bottleneck
        x = self.b40(x, i2)
        x = self.b41(x)
        x = self.b42(x)

        # The fifth bottleneck
        x = self.b50(x, i1)
        x = self.b51(x)


        """"Second Branch"""
        # x_res = self.FA2(x_res)

        # x_res = self.b61(x_res)
        # x_res = self.b62(x_res)
        # x_res, i4 = self.b63(x_res)
        # x_res = self.b64(x_res)
        # x_res = self.b65(x_res)

        # x_res = self.FA3(x_res)

        # x_res = self.b71(x_res)
        # x_res = self.b72(x_res)

        # x_res = self.b73(x_res, i4)
        # x_res = self.b74(x_res)

        # x = x_res

        # x = torch.cat([x, x_res],dim=1)
        x = torch.cat([x, x],dim=1)

        x = self.final(x)

            
        return x


class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels, 
                karnel: int= 3, padding= 1, asymetric = False, 
                dilation: int = None, dropout = 0.2):  
        super(DenseLayer, self).__init__()
        # self.norm = nn.BatchNorm2d(in_channels)
        # self.relu = nn.ReLU(inplace=True)
        self.conv = _DSConv(in_channels,out_channels)
        if dilation:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=dilation,
                                   dilation=dilation, bias=False)
        if asymetric:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=(5, 1), padding=(2, 0), bias=False),
                nn.Conv2d(in_channels, in_channels, kernel_size=(1, 5), padding=(0, 2), bias=False))
        

        # self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.drop =  nn.Dropout2d(p=dropout)

        

    def forward(self, x):
        # x = self.norm(x)
        # x = self.relu(x)
        x = self.conv(x)
        # x = self.conv2(x)
        x = self.drop(x) 
        return x



class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers = 3,
                karnel: int= 3, padding= 1, asymetric = False,
                dilation: int = None, p = 0.2, rate= 16):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        self.channels = in_channels
        self.num_layers = num_layers

        for i in range(num_layers):
            self.layers.append(DenseLayer(in_channels * (2**i), out_channels * (2**i), karnel, padding, asymetric, dilation, p))
        
        output = 0
        for i in range(num_layers):
            output += out_channels * (2**(i))

        self.final = nn.Conv2d(in_channels=output, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        x_list = []
        for i in range(self.num_layers):
            x_res = x
            x = self.layers[i](x)
            x_list.append(x)
            x = torch.cat([x, x_res], 1)
        x = torch.cat(x_list, 1)
        x = self.final(x)
        return x




class FAscnn_pp_V8(nn.Module):
    def __init__(self,in_chanels, num_classes):
        super(FAscnn_pp_V8, self).__init__()
        self.conv =  nn.Conv2d(in_chanels, 16, kernel_size=3, stride=2, padding=1)


        self.b10 = nn.Conv2d(in_channels=16, out_channels=64, dilation=1, kernel_size=3, stride=2, padding=1)
        self.b11 = DenseBlock(in_channels=64, out_channels=64)
        self.b12 = DenseBlock(in_channels=64, out_channels=64)
        self.b13 = DenseBlock(in_channels=64, out_channels=64)
        self.b14 = DenseBlock(in_channels=64, out_channels=64)

        # The second bottleneck
        self.b20 = nn.Conv2d(in_channels=64, out_channels=128, dilation=1, kernel_size=3, stride=2, padding=1)
        self.b21 = DenseBlock(in_channels=128, out_channels=128)
        self.b22 = DenseBlock(in_channels=128, out_channels=128)
        self.b23 = DenseBlock(in_channels=128, out_channels=128, asymetric=True)

        self.b24 = DenseBlock(in_channels=128, out_channels=128)
        self.b25 = DenseBlock(in_channels=128, out_channels=128)

        # The third bottleneck
        self.b31 = DenseBlock(in_channels=128, out_channels=128)
        self.b32 = DenseBlock(in_channels=128, out_channels=128)
        self.b33 = DenseBlock(in_channels=128, out_channels=128)
        self.b34 = DenseBlock(in_channels=128, out_channels=128)
        self.b35 = DenseBlock(in_channels=128, out_channels=128)
        self.b36 = DenseBlock(in_channels=128, out_channels=128)

        # The fourth bottleneck
        self.b40 = nn.ConvTranspose2d(in_channels=128, out_channels=64,kernel_size=3, 
                                           stride=2, padding=1, output_padding=1, bias=False)
        self.b41 = DenseBlock(in_channels=64, out_channels=64)
        self.b42 = DenseBlock(in_channels=64, out_channels=64)
        self.b50 = nn.ConvTranspose2d(in_channels=64, out_channels=16,kernel_size=3, 
                                           stride=2, padding=1, output_padding=1, bias=False)
        self.b51 = DenseBlock(in_channels=16, out_channels=16)



        self.FA1 = FastAttention(128, 32)


        # self.FA2 = FastAttention(16,8)

        # self.b61 = DDBottleNeck(in_channels=16, out_channels=16, dilation=1, downsampling=False, p=0.01)
        # self.b62 = DDBottleNeck(in_channels=16, out_channels=16, dilation=1, downsampling=False, p=0.01)
        # self.b63 = DDBottleNeck(in_channels=16, out_channels=64, dilation=1, downsampling=True, p=0.01)
        # self.b64 = DDBottleNeck(in_channels=64, out_channels=64, dilation=1, downsampling=False, p=0.01)
        # self.b65 = DDBottleNeck(in_channels=64, out_channels=64, dilation=1, downsampling=False, p=0.01)

        # self.FA3 = FastAttention(64,16)



        # self.b71 = DDBottleNeck(in_channels=64, out_channels=64, dilation=1, downsampling=False)
        # self.b72 = DDBottleNeck(in_channels=64, out_channels=64, dilation=1, downsampling=False)
        # self.b73 = UBottleNeck(in_channels=64, out_channels=16)
        # self.b74 = DDBottleNeck(in_channels=16, out_channels=16, dilation=1, downsampling=False)

        # self.FFM = FeatureFusionModulev6(16,16,out_channels=16)

        self.final = nn.ConvTranspose2d(in_channels=32, out_channels=num_classes, kernel_size=3, 
                                           stride=2, padding=1, output_padding=1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        x_res = x
        """"First Branch"""
        x = self.b10(x)
        x = self.b11(x)
        x = self.b12(x)
        x = self.b13(x)
        x = self.b14(x)

        # The second bottleneck
        x = self.b20(x)
        x = self.b21(x)
        x = self.b22(x)
        x = self.b23(x)
        x = self.b24(x)
        x = self.b25(x)


        # The third bottleneck
        x = self.b31(x)
        x = self.b32(x)
        x = self.b33(x)
        x = self.b34(x)
        x = self.b35(x)
        x = self.b36(x)

        x = self.FA1(x)

        # The fourth bottleneck
        x = self.b40(x)
        x = self.b41(x)
        x = self.b42(x)

        # The fifth bottleneck
        x = self.b50(x)
        x = self.b51(x)


        """"Second Branch"""
        # x_res = self.FA2(x_res)

        # x_res = self.b61(x_res)
        # x_res = self.b62(x_res)
        # x_res, i4 = self.b63(x_res)
        # x_res = self.b64(x_res)
        # x_res = self.b65(x_res)

        # x_res = self.FA3(x_res)

        # x_res = self.b71(x_res)
        # x_res = self.b72(x_res)

        # x_res = self.b73(x_res, i4)
        # x_res = self.b74(x_res)

        # x = torch.cat([x, x_res],dim=1)

        x = torch.cat([x, x],dim=1)

        x = self.final(x)

            
        return x




class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = act
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return self.relu(x) if self.act else x


class EfficientAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, HW, C]
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, -1, self.heads, c // self.heads).transpose(1, 2), qkv)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, -1, c)
        out = self.proj(out)
        out = out.transpose(1, 2).reshape(b, c, h, w)
        return out


class MiniSegFormer(nn.Module):
    def __init__(self, num_classes=19):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            ConvBlock(3, 32, stride=2),
            ConvBlock(32, 64, stride=2),
        )

        # Lightweight CNN blocks
        self.cnn_block1 = ConvBlock(64, 128, stride=2)
        self.cnn_block2 = ConvBlock(128, 128)

        # Transformer-based block
        self.attn = EfficientAttention(128, heads=4)

        # Decoder
        self.decoder = nn.Sequential(
            ConvBlock(128, 64),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBlock(64, 32),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.cnn_block1(x)
        x = self.cnn_block2(x)
        x = self.attn(x)
        x = self.decoder(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # final upsampling
        return x

class _ConvBNReLUv9(nn.Module):
    """Conv-BN-ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, **kwargs):
        super(_ConvBNReLUv9, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class _DSConv9(nn.Module):
    """Depthwise Separable Convolutions"""

    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(_DSConv9, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, dw_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(dw_channels),
            nn.ReLU(True),
            nn.Conv2d(dw_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class _DWConv9(nn.Module):
    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(_DWConv9, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, out_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class LinearBottleneckv9(nn.Module):
    """LinearBottleneck used in MobileNetV2"""

    def __init__(self, in_channels, out_channels, t=6, stride=2, **kwargs):
        super(LinearBottleneckv9, self).__init__()
        self.use_shortcut = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(
            # pw
            _ConvBNReLUv9(in_channels, in_channels * t, 1),
            # dw
            _DWConv9(in_channels * t, in_channels * t, stride),
            # pw-linear
            nn.Conv2d(in_channels * t, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.block(x)
        if self.use_shortcut:
            out = x + out
        return out


class PyramidPoolingv9(nn.Module):
    """Pyramid pooling module"""

    def __init__(self, in_channels, out_channels, **kwargs):
        super(PyramidPoolingv9, self).__init__()
        inter_channels = int(in_channels / 4)
        self.conv1 = _ConvBNReLUv9(in_channels, inter_channels, 1, **kwargs)
        self.conv2 = _ConvBNReLUv9(in_channels, inter_channels, 1, **kwargs)
        self.conv3 = _ConvBNReLUv9(in_channels, inter_channels, 1, **kwargs)
        self.conv4 = _ConvBNReLUv9(in_channels, inter_channels, 1, **kwargs)
        self.out = _ConvBNReLUv9(in_channels * 2, out_channels, 1)

    def pool(self, x, size):
        avgpool = nn.AdaptiveAvgPool2d(size)
        return avgpool(x)

    def upsample(self, x, size):
        return F.interpolate(x, size, mode='bilinear', align_corners=True)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = self.upsample(self.conv1(self.pool(x, 1)), size)
        feat2 = self.upsample(self.conv2(self.pool(x, 2)), size)
        feat3 = self.upsample(self.conv3(self.pool(x, 3)), size)
        feat4 = self.upsample(self.conv4(self.pool(x, 6)), size)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = self.out(x)
        return x


class LearningToDownsamplev9(nn.Module):
    """Learning to downsample module"""

    def __init__(self, in_channels ,dw_channels1=32, dw_channels2=48, out_channels=64, **kwargs):
        super(LearningToDownsamplev9, self).__init__()
        self.conv = _ConvBNReLUv9(in_channels, dw_channels1, 3, 2)
        self.dsconv1 = _DSConv9(dw_channels1, dw_channels2, 2)
        self.dsconv2 = _DSConv9(dw_channels2, out_channels, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        return x


class GlobalFeatureExtractorv9(nn.Module):
    """Global feature extractor module"""

    def __init__(self, in_channels=64, block_channels=(64, 96, 128),
                 out_channels=128, t=6, num_blocks=(3, 3, 3), **kwargs):
        super(GlobalFeatureExtractorv9, self).__init__()
        self.bottleneck1 = self._make_layer(LinearBottleneckv9, in_channels, block_channels[0], num_blocks[0], t, 2)
        self.bottleneck2 = self._make_layer(LinearBottleneckv9, block_channels[0], block_channels[1], num_blocks[1], t, 2)
        self.bottleneck3 = self._make_layer(LinearBottleneckv9, block_channels[1], block_channels[2], num_blocks[2], t, 1)
        self.ppm = PyramidPoolingv9(block_channels[2], out_channels)

    def _make_layer(self, block, inplanes, planes, blocks, t=6, stride=1):
        layers = []
        layers.append(block(inplanes, planes, t, stride))
        for i in range(1, blocks):
            layers.append(block(planes, planes, t, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.ppm(x)
        return x


class FeatureFusionModulev9(nn.Module):
    """Feature fusion module"""

    def __init__(self, highter_in_channels, lower_in_channels, out_channels, scale_factor=4, **kwargs):
        super(FeatureFusionModulev9, self).__init__()
        self.scale_factor = scale_factor
        self.dwconv = _DWConv9(lower_in_channels, out_channels, 1)
        self.conv_lower_res = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
        self.conv_higher_res = nn.Sequential(
            nn.Conv2d(highter_in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(True)

    def forward(self, higher_res_feature, lower_res_feature):
        lower_res_feature = F.interpolate(lower_res_feature, scale_factor=4, mode='bilinear', align_corners=True)
        lower_res_feature = self.dwconv(lower_res_feature)
        lower_res_feature = self.conv_lower_res(lower_res_feature)

        higher_res_feature = self.conv_higher_res(higher_res_feature)
        out = higher_res_feature + lower_res_feature
        return self.relu(out)


class Classiferv9(nn.Module):
    """Classifer"""

    def __init__(self, dw_channels, num_classes, stride=1, **kwargs):
        super(Classiferv9, self).__init__()
        self.dsconv1 = _DSConv9(dw_channels, dw_channels, stride)
        self.dsconv2 = _DSConv9(dw_channels, dw_channels, stride)
        self.conv = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(dw_channels, num_classes, 1)
        )

    def forward(self, x):
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = self.conv(x)
        return x



class FAscnn_pp_V9(nn.Module):
    def __init__(self, in_channels, num_classes, aux=False, **kwargs):
        super(FAscnn_pp_V9, self).__init__()
        self.aux = aux
        self.learning_to_downsample = LearningToDownsamplev9(in_channels,32, 48, 64)
        self.FA1 = FastAttention(64,16)
        self.global_feature_extractor = GlobalFeatureExtractorv9(64, [64, 96, 128], 128, 6, [3, 3, 3])
        self.FA2 = FastAttention(128,32)
        self.feature_fusion = FeatureFusionModulev9(64, 128, 128)
        self.classifier = Classiferv9(128, num_classes)
        if self.aux:
            self.auxlayer = nn.Sequential(
                nn.Conv2d(64, 32, 3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Dropout(0.1),
                nn.Conv2d(32, num_classes, 1)
            )

    def forward(self, x):
        size = x.size()[2:]
        higher_res_features = self.learning_to_downsample(x)
        x = self.global_feature_extractor(higher_res_features)
        higher_res_features = self.FA1(higher_res_features)
        x = self.FA2(x)
        x = self.feature_fusion(higher_res_features, x)
        x = self.classifier(x)
        outputs = []
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        outputs.append(x)
        if self.aux:
            auxout = self.auxlayer(higher_res_features)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
            outputs.append(auxout)
            return tuple(outputs)
        return x
    

class Residual_Blockv10(nn.Module):
    def __init__(self,in_channel, num_conv= 3): # out chanells is the same as input
        super().__init__()
        self.conv = nn.ModuleList()
        self.FA = nn.ModuleList()
        self.relu = nn.ReLU(inplace=False)
        self.norm = nn.BatchNorm2d(in_channel)
        self.num_conv = num_conv
        for i in range(self.num_conv):
            self.conv.append(nn.Conv2d(in_channel,in_channel,3, padding=1, bias=False))
            self.FA.append(FastAttention(in_channel,int(in_channel / (2**(i+1)))))


    def forward(self, x):
        for i in range(self.num_conv):
            x_res = x
            x = self.conv[i](x)
            x = self.norm(x)
            x = self.relu(x)
            x_res = self.FA[i](x_res)
            x = x + x_res

        x = self.norm(x)
        x = self.relu(x)

        return x



class _ConvBNReLUv10(nn.Module):
    """Conv-BN-ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3,padding=None,  dilation=1,bias = False,):
        super(_ConvBNReLUv10, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=dilation,dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)




class FAscnn_pp_V10(nn.Module):
    def __init__(self, in_channels, num_classes, aux=False, **kwargs):
        super(FAscnn_pp_V10, self).__init__()
        self.relu = nn.ReLU(inplace=False)

        self.inicial = nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1)
        """First branch"""
        #encoder
        self.dc1 = _ConvBNReLUv10(16, 16, kernel_size=3, padding=4,
                                   dilation=4, bias=False)
        self.conv1 = nn.Conv2d(16,64,1)

        # self.dc2 = _ConvBNReLUv10(32, 32, kernel_size=3, padding=4,
        #                            dilation=4, bias=False)
        # self.conv2 = nn.Conv2d(32,64,1)

        self.dc3 = _ConvBNReLUv10(64, 64, kernel_size=3, padding=4,
                                   dilation=4, bias=False)

        self.FA1 = FastAttention(16,8)
        self.FA2 = FastAttention(32,16)
        self.FA3 = FastAttention(64,32)


        self.residual_blocks1 = nn.Sequential(
            Residual_Blockv10(64)
        )

        #encoder

        self.conv_pool1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ConvTranspose2d(in_channels=64,out_channels=32, kernel_size=3, 
                                           stride=2, padding=1, output_padding=1, bias=False)
        )

        self.conv_pool2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ConvTranspose2d(in_channels=32,out_channels=16, kernel_size=3, 
                                           stride=2, padding=1, output_padding=1, bias=False)
        )



        


        """Second branch"""

        #decoder
        self.dc4 = _ConvBNReLUv10(16, 16, kernel_size=3, padding=4,
                                   dilation=4, bias=False)
        self.conv4 = nn.Conv2d(16,64,1)

        # self.dc5 = _ConvBNReLUv10(32, 32, kernel_size=3, padding=4,
        #                            dilation=4, bias=False)
        # self.conv5 = nn.Conv2d(32,64,1)

        self.dc6 = _ConvBNReLUv10(64, 64, kernel_size=3, padding=4,
                                   dilation=4, bias=False)
        self.conv6 = nn.Conv2d(64,128,1)

        self.dc7 = _ConvBNReLUv10(128, 128, kernel_size=3, padding=4,
                                   dilation=4, bias=False)
        
    
        self.FA4 = FastAttention(16,8)
        self.FA5 = FastAttention(32,16)
        self.FA6 = FastAttention(64,32)

        self.residual_blocks2 = nn.Sequential(
            Residual_Blockv10(128)
        )

        # encoder

        self.conv_pool3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ConvTranspose2d(in_channels=128,out_channels=64, kernel_size=3, 
                                           stride=2, padding=1, output_padding=1, bias=False)
        )

        self.conv_pool4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ConvTranspose2d(in_channels=64,out_channels=32, kernel_size=3, 
                                           stride=2, padding=1, output_padding=1, bias=False)
        )

        self.conv_pool5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ConvTranspose2d(in_channels=32,out_channels=16, kernel_size=3, 
                                           stride=2, padding=1, output_padding=1, bias=False)
        )



        self.final =  nn.ConvTranspose2d(in_channels=16, out_channels=num_classes, kernel_size=3, 
                                           stride=2, padding=1, output_padding=1, bias=False)


    def forward(self, x):
        x = self.inicial(x)
        x_res = x
        x = self.dc1(x)
        x_res1 = x
        x = self.conv1(x)
        # x = self.dc2(x)
        # x_res2 = x
        # x = self.conv2(x)
        x = self.dc3(x)
        x_res3 = x
        x = self.residual_blocks1(x)



        x_res = self.dc4(x_res)
        x_res4 = x_res
        x_res = self.conv4(x_res)
        # x_res = self.dc5(x_res)
        # x_res5 = x_res
        # x_res = self.conv5(x_res)
        x_res = self.dc6(x_res)
        x_res6 = x_res
        x_res = self.conv6(x_res)
        x_res = self.dc7(x_res)
        x_res = self.residual_blocks2(x_res)

        x = x + self.FA6(x_res3)
        x = self.conv_pool1(x)
        # x = x + self.FA5(x_res2)
        x = self.conv_pool2(x)
        x = x + self.FA4(x_res1)
        
        x_res = self.conv_pool3(x_res)
        x_res = x_res + self.FA3(x_res6)
        x_res = self.conv_pool4(x_res)
        # x_res = x_res + self.FA2(x_res5)
        x_res = self.conv_pool5(x_res)
        x_res = x_res + self.FA1(x_res4)




        x = x + x_res
        x = self.final(x)
        return x
    
    

class FAscnn_pp_V11(nn.Module):
    def __init__(self, in_channels, num_classes, aux=False):
        super(FAscnn_pp_V11, self).__init__()
        

    def forward(self, x):
        return self.model(x)



if __name__ == '__main__':



    def extract_patches(x, tile_size, overlap):
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        th, tw = tile_size
        oh, ow = overlap

        sh, sw = th - oh, tw - ow

        patches = []
        coords = []

        ys = list(range(0, H - th + 1, sh))
        xs = list(range(0, W - tw + 1, sw))


        if ys[-1] != H - th:
            ys.append(H - th)
        if xs[-1] != W - tw:
            xs.append(W - tw)

        for y in ys:
            for x0 in xs:
                patches.append(x[:, :, y:y+th, x0:x0+tw])
                coords.append((y, x0))
        
        patches = torch.cat(patches, dim=0)  # (N*B, C, th, tw)
        return patches, coords



    def merge_patches(patches, coords, out_shape, tile_size):
        """
        patches: (N, C, th, tw)
        out_shape: (B, C, H, W)
        """
        B, C, H, W = out_shape
        th, tw = tile_size

        output = torch.zeros(out_shape, device=patches.device)
        counter = torch.zeros(out_shape, device=patches.device)

        idx = 0
        for y, x in coords:
            output[:, :, y:y+th, x:x+tw] += patches[idx:idx+B]
            counter[:, :, y:y+th, x:x+tw] += 1
            idx += B

        return output / counter.clamp(min=1)


    class FAscnn_pp_Patched(nn.Module):
            def __init__(
                self,
                base_model: nn.Module,
                tile_size=(256, 256),
                overlap=(64, 64)
            ):
                super().__init__()
                self.model = base_model
                self.tile_size = tile_size
                self.overlap = overlap

            def forward(self, x):
                """
                x: (B, C, H, W)
                """
                B, C, H, W = x.shape

                patches, coords = extract_patches(
                    x, self.tile_size, self.overlap
                )

                # inference na patchach
                out = []
                for parch in patches:
                    patch_out = self.model(parch)
                    out.append(patch_out)
                out = torch.stack(out, dim=0)
                _, C_out, th, tw = out.shape

                output = merge_patches(
                    torch.cat(out, dim=0),
                    coords,
                    (B, C_out, H, W),
                    (th, tw)
                )

                return output
            def __init__(
                self,
                base_model: nn.Module,
                tile_size=(64, 64),
                overlap=(16, 16)
            ):
                super().__init__()
                self.model = base_model
                self.tile_size = tile_size
                self.overlap = overlap

            def forward(self, x):
                """
                x: (B, C, H, W)
                """
                B, C, H, W = x.shape

                patches, coords = extract_patches(
                    x, self.tile_size, self.overlap
                )

                # inference na patchach
                patch_out = self.model(patches)
                _, C_out, th, tw = patch_out.shape

                output = merge_patches(
                    patch_out,
                    coords,
                    (B, C_out, H, W),
                    (th, tw)
                )

                return output

    # img = torch.randn(1, 3, 256, 512)
    # img = torch.randn(1, 3, 360, 640)
    # img = torch.randn(1, 3, 720, 1280)
    # img = torch.randn(1, 3, 864,1600)
    
    img = torch.randn(1, 3, 512, 512)


    print('cpu')

    base_model = FAscnn_pp_V6(3,1)
    
    model = FAscnn_pp_Patched(
    base_model,
    tile_size=(64, 64),
    overlap=(8, 8)
    )
    
    # model = FAscnn_pp_V6(3,1)


    model.eval()
    start = time.perf_counter()
    outputs = model(img)
    end = time.perf_counter()

    print(outputs.shape)



    elapsed_ms = (end - start) * 1000
    print(f"Czas wykonania: {elapsed_ms:.3f} ms")
    fps = 1 / (end - start)
    print(f"FPS: {fps:.2f}")


    flops, params = profile(model, inputs=(img,))
    print(f"FLOPs: {flops/1e9:.2f} GFLOPs")
    print(f"Params: {params/1e6:.2f} M")

