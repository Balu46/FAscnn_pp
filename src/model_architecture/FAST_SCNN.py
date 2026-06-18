
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from thop import profile



"""Fast Segmentation Convolutional Neural Network"""
# https://github.com/Tramac/Fast-SCNN-pytorch
class FastSCNN(nn.Module):
    def __init__(self, num_classes, aux=True, **kwargs):
        super(FastSCNN, self).__init__()
        self.aux = aux
        self.learning_to_downsample = LearningToDownsample(32, 48, 64)
        self.global_feature_extractor = GlobalFeatureExtractor(64, [64, 96, 128], 128, 6, [3, 3, 3])
        self.feature_fusion = FeatureFusionModule(64, 128, 128)
        self.classifier = Classifer(128, num_classes)
        if self.aux:
            # self.auxlayer = nn.Sequential(
            #     nn.Conv2d(64, 32, 3, padding=1, bias=False),
            #     nn.BatchNorm2d(32),
            #     nn.ReLU(True),
            #     nn.Dropout(0.1),
            #     nn.Conv2d(32, num_classes, 1)
            # )
            self.aux_downsaple = nn.Sequential(
                nn.Conv2d(64, 32, 3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Dropout(0.1),
                nn.Conv2d(32, num_classes, 1)
            )
            self.aux_global = nn.Sequential(
                nn.Conv2d(128, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.Dropout(0.1),
                nn.Conv2d(64, num_classes, 1)
            )
            

    def forward(self, x):
        size = x.size()[2:]
        higher_res_features, _ = self.learning_to_downsample(x)
        context = self.global_feature_extractor(higher_res_features)
        x = self.feature_fusion(higher_res_features, context)
        x = self.classifier(x)
        outputs = []
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        outputs.append(x)
        
        # if self.training:
        #     auxout = self.auxlayer(higher_res_features)
        #     auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
        #     outputs.append(auxout)
        #     return tuple(outputs)
        
        if self.training:
            aux_spatial = self.aux_downsaple(higher_res_features)
            aux_spatial = F.interpolate(aux_spatial, size=size, mode='bilinear', align_corners=False)
            
            aux_context = self.aux_global(context)
            aux_context = F.interpolate(aux_context, size=size, mode='bilinear', align_corners=False)
            
            return {
                "main": x,
                "context": aux_context,
                "spatial": aux_spatial
            }
        
        return x
    
    def __type__(self):
        return "FastSCNN"




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

    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(_DSConv, self).__init__()
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


class _DWConv(nn.Module):
    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(_DWConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, out_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class LinearBottleneck(nn.Module):
    """LinearBottleneck used in MobileNetV2"""

    def __init__(self, in_channels, out_channels, t=6, stride=2, **kwargs):
        super(LinearBottleneck, self).__init__()
        self.use_shortcut = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(
            # pw
            _ConvBNReLU(in_channels, in_channels * t, 1),
            # dw
            _DWConv(in_channels * t, in_channels * t, stride),
            # pw-linear
            nn.Conv2d(in_channels * t, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.block(x)
        if self.use_shortcut:
            out = x + out
        return out


class PyramidPooling(nn.Module):
    """Pyramid pooling module"""

    def __init__(self, in_channels, out_channels, **kwargs):
        super(PyramidPooling, self).__init__()
        inter_channels = int(in_channels / 4)
        self.conv1 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv2 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv3 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv4 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.out = _ConvBNReLU(in_channels * 2, out_channels, 1)

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


class LearningToDownsample(nn.Module):
    """Learning to downsample module"""

    def __init__(self, dw_channels1=32, dw_channels2=48, out_channels=64, **kwargs):
        super(LearningToDownsample, self).__init__()
        self.conv = _ConvBNReLU(3, dw_channels1, 3, 2)
        self.dsconv1 = _DSConv(dw_channels1, dw_channels2, 2)
        self.dsconv2 = _DSConv(dw_channels2, out_channels, 2)

    def forward(self, x):
        x = self.conv(x)
        x_h4 = self.dsconv1(x)
        x = self.dsconv2(x_h4)
        return x, x_h4


class GlobalFeatureExtractor(nn.Module):
    """Global feature extractor module"""

    def __init__(self, in_channels=64, block_channels=(64, 96, 128),
                 out_channels=128, t=6, num_blocks=(3, 3, 3), **kwargs):
        super(GlobalFeatureExtractor, self).__init__()
        self.bottleneck1 = self._make_layer(LinearBottleneck, in_channels, block_channels[0], num_blocks[0], t, 2)
        self.bottleneck2 = self._make_layer(LinearBottleneck, block_channels[0], block_channels[1], num_blocks[1], t, 2)
        self.bottleneck3 = self._make_layer(LinearBottleneck, block_channels[1], block_channels[2], num_blocks[2], t, 1)
        self.ppm = PyramidPooling(block_channels[2], out_channels)

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


class FeatureFusionModule(nn.Module):
    """Feature fusion module"""

    def __init__(self, highter_in_channels, lower_in_channels, out_channels, scale_factor=4, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.scale_factor = scale_factor
        self.dwconv = _DWConv(lower_in_channels, out_channels, 1)
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


class Classifer(nn.Module):
    """Classifer"""

    def __init__(self, dw_channels, num_classes, stride=1, **kwargs):
        super(Classifer, self).__init__()
        self.dsconv1 = _DSConv(dw_channels, dw_channels, stride)
        self.dsconv2 = _DSConv(dw_channels, dw_channels, stride)
        self.conv = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(dw_channels, num_classes, 1)
        )

    def forward(self, x):
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = self.conv(x)
        return x




 

if __name__ == '__main__':
    import os
    import time
    import numpy as np
    from thop import profile
    
    def switch_model_to_deploy(module: nn.Module):
    # Idziemy po dzieciach, a nie po wszystkich modułach naraz
        for m in module.children():
            if hasattr(m, "switch_to_deploy"):
                m.switch_to_deploy()
            switch_model_to_deploy(m) # Rekurencja po drzewie


    def model_size_mb(model):
        return sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    if device.type == 'cpu':
        torch.set_num_threads(os.cpu_count())
        torch.set_num_interop_threads(1)

    img = torch.randn(1, 3, 1024, 2048).to(device)

    print(device)

    model = FastSCNN(3, 19).to(device)
    switch_model_to_deploy(model)  # Przełączamy wszystkie RepConv na tryb deploy (fuzja wag) przed testem wydajności 
    model.eval()


    print("Model :", model.__type__())

    warmup_iters = 50
    timed_iters = 200

    with torch.inference_mode():
        for _ in range(warmup_iters):
            _ = model(img)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        times = []
        for _ in range(timed_iters):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(img)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000.0)  # ms

    times = np.array(times)
    mean_ms = times.mean()
    # p50_ms = np.percentile(times, 50)
    # p95_ms = np.percentile(times, 95)
    # p99_ms = np.percentile(times, 99)
    fps = 1000.0 / mean_ms

    print(f"Latency mean: {mean_ms:.3f} ms")
    # print(f"Latency p50 : {p50_ms:.3f} ms")
    # print(f"Latency p95 : {p95_ms:.3f} ms")
    # print(f"Latency p99 : {p99_ms:.3f} ms")
    print(f"FPS         : {fps:.2f}")

    # with torch.inference_mode():
    #     outputs = model(img)
    # print(outputs.shape)

    flops, params = profile(model, inputs=(img,))
    print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
    print(f"Params: {params / 1e6:.2f} M")
    print(f"Model size: {model_size_mb(model):.2f} MB")