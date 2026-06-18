import torch
import torch.nn as nn
# from enet_modules import InitialBlock, DDBottleNeck, ABottleNeck, UBottleNeck
import  time
import torch.nn.functional as F
import os
import sys
from thop import profile

from audioop import bias
from numpy import indices
import torch
import torch.nn as nn

class InitialBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=13):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.activation = nn.PReLU(16)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        main = self.conv(x)
        main = self.norm(main)
        
        side = self.max_pool(x)

        x = torch.cat((main, side), dim=1)
        x = self.activation(x)
        return x

class DDBottleNeck(nn.Module):
    """
    Bottleneck 2.x for dowmsampling and dilated types
    """
    def __init__(self, in_channels, out_channels, dilation, downsampling, activation='prelu', down_ratio=4, p=0.1):
        super().__init__()

        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.dilation     = dilation
        self.downsampling = downsampling

        if self.downsampling: # if bottleneck 2.x with downsampling
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
        self.conv2 = nn.Conv2d(in_channels=self.down_channels, out_channels=self.down_channels, kernel_size=3, stride=self.stride, \
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
            temp = torch.zeros((batch_size, out_shape, x.shape[2], x.shape[3]), device=x_main.device)
            # if torch.cuda.is_available():
            #     temp = temp.cuda()
            
            x_main = torch.cat((x_main, temp), dim=1)

        # Side + Main
        x = x + x_main
        x = self.activation3(x)

        if self.downsampling:
            return x, indices
        else:
            return x

class ABottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, activation='prelu', down_ratio=4):
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

        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.down_channels, kernel_size=1, stride=1,\
                                padding=0, bias=False)
        self.activation1 = self.activation

        self.conv21 = nn.Conv2d(in_channels=self.down_channels, out_channels=self.down_channels, kernel_size=(1, 5), stride=1,\
                                padding=(0, 2), bias=False)
        self.conv22 = nn.Conv2d(in_channels=self.down_channels, out_channels=self.down_channels, kernel_size=(5, 1), stride=1,\
                                padding=(2, 0), bias=False)
        self.activation2 = self.activation

        self.conv3 = nn.Conv2d(in_channels=self.down_channels, out_channels=self.out_channels, kernel_size=1, stride=1,\
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
            temp = torch.zeros((batch_size, out_shape, x.shape[2], x.shape[3]), device=x_main.device)
            # if torch.cuda.is_available():
            #     temp = temp.cuda()
            x_main = torch.cat((x_main, temp), dim=1)

        # Side + Main
        x += x_main
        x = self.activation3(x)
        return x

class UBottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, activation='prelu', down_ratio=4):
        super().__init__()

        self.in_channels   = in_channels
        self.out_channels  = out_channels
        self.down_channels = int(self.in_channels / down_ratio)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'mish':
            self.activation = nn.Mish()

        self.main_conv   = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1)
        
        self.convt1      = nn.ConvTranspose2d(in_channels=self.in_channels, out_channels=self.down_channels, kernel_size=1, padding=0, bias=False)
        self.activation1 = self.activation
        self.convt2      = nn.ConvTranspose2d(in_channels=self.down_channels, out_channels=self.down_channels, kernel_size=3, stride=2, padding=1,\
                                            output_padding=1, bias=False)
        self.activation2 = self.activation
        self.convt3      = nn.ConvTranspose2d(in_channels=self.down_channels, out_channels=self.out_channels, kernel_size=1, padding=0, bias=False)
        self.activation3 = self.activation

        self.norm1       = nn.BatchNorm2d(self.down_channels)
        self.norm2       = nn.BatchNorm2d(self.down_channels)
        self.norm3       = nn.BatchNorm2d(self.out_channels)

        self.unpool     = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.drop_out    = nn.Dropout2d(p=0.1)

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
        x_main = self.unpool(x_main, indices, output_size=x.size())

        # Concatenate
        x = x + x_main
        x  = self.activation3(x)

        return x

# https://github.com/ntkhoa95/ENet_PyTorch?tab=MIT-1-ov-file
class ENet(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        self.C = n_class  # number of classes

        # The Initial block
        self.initial = InitialBlock()

        # The first bottleneck
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

        self.fullconv = nn.ConvTranspose2d(in_channels=16, out_channels=self.C, kernel_size=3, 
                                           stride=2, padding=1, output_padding=1, bias=False)

    def forward(self, x):
        # The initial block
        x = self.initial(x)

        # The first bottleneck
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

        # The fourth bottleneck
        x = self.b40(x, i2)
        x = self.b41(x)
        x = self.b42(x)

        # The fifth bottleneck
        x = self.b50(x, i1)
        x = self.b51(x)

        # Final ConvTranspose Layer
        x = self.fullconv(x)

        return x

    def __type__(self):
        return "ENet_original"


if __name__ == '__main__':


    # img = torch.randn(1, 3, 256, 512)
    img = torch.randn(1, 3, 360, 640)
    # img = torch.randn(1, 3, 720, 1280)
    # img = torch.randn(1, 3, 864,1600)

    device = torch.device('cpu')


    print('cpu')

    model = ENet(1).to(device)
    model.eval()
    start = time.perf_counter()
    outputs = model(img.to(device))
    end = time.perf_counter()

    print(outputs.shape)




    elapsed_ms = (end - start) * 1000
    print(f"Czas wykonania: {elapsed_ms:.3f} ms")
    fps = 1 / (end - start)
    print(f"FPS: {fps:.2f}")


    flops, params = profile(model, inputs=(img,))
    print(f"FLOPs: {flops/1e9:.2f} GFLOPs")
    print(f"Params: {params/1e6:.2f} M")
