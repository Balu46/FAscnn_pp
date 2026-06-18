from thop import profile
import torch
import torch.nn as nn
import  time
import torch.nn.functional as F
import os
import sys
# utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils'))
# sys.path.append(utils_path)
# from usefull import *


# Initial Block
class InitialBlock(nn.Module):
    def __init__(self, in_chanels = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_chanels, 13, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        conv_out = self.conv(x)
        pool_out = self.pool(x)
        return torch.cat([conv_out, pool_out], dim=1)  # (13+3 = 16 kanałów)


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


# ENet model 
class ENetv2(nn.Module):
    def __init__(self,num_classes, in_channels=3):
        super().__init__()
        self.initial = InitialBlock(in_channels)

        # Stage 1
        self.bottleneck1 = nn.Sequential(
            Bottleneck(16, 64, downsample=True),
            *[Bottleneck(64, 64) for _ in range(4)],
        )

        # Stage 2
        self.bottleneck2 = nn.Sequential(
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

        # Decoder - uproszczony
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.deconv2 = nn.ConvTranspose2d(64, 16, kernel_size=2, stride=2)
        self.classifier = nn.ConvTranspose2d(16, num_classes, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.initial(x)
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.classifier(x)
        return x  # (B, num_classes, H, W)

    def __type__(self):
        return "ENet_v2"





class InitialBlock_v3(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x):
        return self.conv(x)

# Bottleneck Block - GroupNorm
class Bottleneck_v3(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False, dilated=1, asymmetric=False, dropout_prob=0.3):
        super().__init__()
        internal_channels = in_channels // 8

        self.downsample = downsample
        self.dropout_prob = dropout_prob

        self.conv1 = nn.Conv2d(in_channels, internal_channels, kernel_size=1, bias=False)
        self.ln1 = nn.GroupNorm(1, internal_channels)  # 1 grupa = LayerNorm per channel
        self.prelu1 = nn.PReLU()

        if downsample:
            self.conv2 = nn.Conv2d(internal_channels, internal_channels, kernel_size=3,groups=internal_channels, stride=2, padding=1, bias=False)
        elif dilated > 1:
            self.conv2 = nn.Conv2d(internal_channels, internal_channels, kernel_size=3, padding=dilated,
                                   dilation=dilated, bias=False)
        else:
            self.conv2 = nn.Conv2d(internal_channels, internal_channels, kernel_size=3,groups=internal_channels, padding=1, bias=False)

        self.ln2 = nn.GroupNorm(1, internal_channels)  # 1 grupa = LayerNorm per channel
        self.prelu2 = nn.PReLU()

        self.conv3 = nn.Conv2d(internal_channels, out_channels, kernel_size=1, bias=False)
        self.ln3 = nn.GroupNorm(1, out_channels)  # 1 grupa = LayerNorm per channel

        self.dropout = nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else nn.Identity()

        self.match_dims = (in_channels != out_channels or downsample)
        if self.match_dims:
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2 if downsample else 1, bias=False)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        
        out = self.ln1(out)
        out = self.prelu1(out)

        out = self.conv2(out)
        out = self.ln2(out)
        out = self.prelu2(out)

        out = self.conv3(out)
        out = self.ln3(out)
        out = self.dropout(out)

        if self.match_dims:
            residual = self.proj(residual)

        out += residual
        return F.relu(out)

# ENet model - UpSampling + Conv2d zamiast ConvTranspose
class ENetv3(nn.Module):
    def __init__(self, num_classes, in_channels=3):
        super().__init__()
        self.initial = InitialBlock_v3(in_channels)

        self.bottleneck1 = nn.Sequential(
            Bottleneck_v3(16, 64, downsample=True),
            # Bottleneck_v3(64, 64),
            Bottleneck_v3(64, 128, downsample=True),
            # Bottleneck_v3(128, 128),
            Bottleneck_v3(128, 128, dilated=2),
            # Bottleneck_v3(128, 128),
            Bottleneck_v3(128, 128, dilated=4)
        )

        # self.bottleneck2 = nn.Sequential(
        #     Bottleneck_v3(64, 128, downsample=True),
        #     Bottleneck_v3(128, 128),
        #     Bottleneck_v3(128, 128, dilated=2),
        #     Bottleneck_v3(128, 128),  # usunięty asymmetric
        #     Bottleneck_v3(128, 128, dilated=4),
        # )

        # Dekoder z Upsample bilinear + Conv2d

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(64, 16, kernel_size=2, stride=2)
        self.final = nn.ConvTranspose2d(16, num_classes, kernel_size=2, stride=2)



        # self.up1 = nn.Sequential(
        #     nn.Upsample(scale_factor=2, mode='bilinear'),
        #     nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
        #     nn.ReLU(inplace=True),
        # )

        # self.up2 = nn.Sequential(
        #     nn.Upsample(scale_factor=2, mode='bilinear'),
        #     nn.Conv2d(64, 16, kernel_size=3, padding=1, bias=False),
        #     nn.ReLU(inplace=True),
        # )

        # self.final = nn.Sequential(
        #     nn.Upsample(scale_factor=2, mode='bilinear'),
        #     nn.Conv2d(16, num_classes, kernel_size=3, padding=1, bias=False)
        # )

    def forward(self, x):
        x = self.initial(x)
        x = self.bottleneck1(x)
        # x = self.bottleneck2(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.final(x)
        return x
    
    def __type__(self):
        return "ENet_v3"






if __name__ == '__main__':


    # img = torch.randn(1, 3, 256, 512)
    # img = torch.randn(1, 3, 360, 640)
    # img = torch.randn(1, 3, 720, 1280)
    # img = torch.randn(1, 3, 864,1600)
    
    img = torch.randn(1, 3, 512, 512)

    print('cpu')

    model = ENetv3(1)
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

