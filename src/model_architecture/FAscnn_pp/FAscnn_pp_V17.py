from xml.parsers.expat import model

from numpy import size
import torch
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import Tensor
import math

# ================================= DEBUGGING ================================= 
def _debug_gate(self, gate, s, c, fused, image=None):
    import os
    import matplotlib.pyplot as plt
    import torch

    save_dir = getattr(self, "debug_dir", ".")
    os.makedirs(save_dir, exist_ok=True)

    print("gate mean:", gate.mean().item())
    print("gate std :", gate.std().item())
    print("gate min :", gate.min().item())
    print("gate max :", gate.max().item())

    def feature_norm(x):
        return x.norm(dim=1).mean().item()

    print("S:", feature_norm(s))
    print("C:", feature_norm(c))
    print("FUSED:", feature_norm(fused))

    gate_map = gate[0].mean(0).detach().cpu().numpy()
    ctx_allow_map = (1.0 - gate[0].mean(0)).detach().cpu().numpy()

    fig = plt.figure(figsize=(15, 4))

    if image is not None:
        if image.dim() == 4:
            img = image[0]
        else:
            img = image

        img = img.detach().cpu().permute(1, 2, 0)

        # opcjonalna denormalizacja pod ImageNet stats:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3)
        img = img * std + mean
        img = img.clamp(0, 1).numpy()

        plt.subplot(1, 3, 1)
        plt.title("Image")
        plt.imshow(img)
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title("Edge gate")
        plt.imshow(gate_map, cmap="jet")
        plt.colorbar()
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.title("Context allow = 1-gate")
        plt.imshow(ctx_allow_map, cmap="jet")
        plt.colorbar()
        plt.axis("off")
    else:
        plt.subplot(1, 2, 1)
        plt.title("Edge gate")
        plt.imshow(gate_map, cmap="jet")
        plt.colorbar()
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Context allow = 1-gate")
        plt.imshow(ctx_allow_map, cmap="jet")
        plt.colorbar()
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "gate_maps.png"))
    plt.close(fig)

    fig = plt.figure(figsize=(6, 4))
    plt.hist(gate.detach().cpu().numpy().flatten(), bins=50)
    plt.title("Gate distribution")
    plt.xlabel("gate value")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "gate_distribution.png"))
    plt.close(fig)

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



class InitialBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=16):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.normalize = nn.BatchNorm2d(out_channels)
        self.act = nn.PReLU()

    def forward(self, x):
        x =  self.conv(x)
        x = self.normalize(x)
        return self.act(x)

class BiSeNetFFM(nn.Module):
    """
    Feature Fusion Module inspirowany BiSeNet:
    - concat spatial + context
    - BN / lokalna projekcja
    - global average pooling
    - channel re-weighting jak SE
    - residual-style boost: y = feat + feat * att
    """
    def __init__(self, spatial_in: int, context_in: int, out_channels: int, reduction: int = 4):
        super().__init__()
        fusion_in = spatial_in + context_in
        hidden = max(out_channels // reduction, 8)

        self.convblk = nn.Sequential(
            nn.Conv2d(fusion_in, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, spatial_feat, context_feat):
        # spatial_feat, context_feat: ten sam H,W
        feat = torch.cat([spatial_feat, context_feat], dim=1)
        feat = self.convblk(feat)
        att = self.attention(feat)
        out = feat + feat * att
        return out

class LearningToDownsampleV2(nn.Module):
    """
    Spatial path / learning-to-downsample:
    1/1 -> 1/2 -> 1/4 -> 1/8
    """
    def __init__(self, in_channels=3, c1=32, c2=48, c3=64):
        super().__init__()
        self.conv = _ConvBNReLU(in_channels, c1, kernel_size=3, stride=2, padding=1)
        self.dsconv1 = _DSConv(c1, c2, stride=2)
        self.dsconv2 = _DSConv(c2, c3, stride=2)

    def forward(self, x):
        x = self.conv(x)      # H/2
        x = self.dsconv1(x)   # H/4
        x = self.dsconv2(x)   # H/8
        return x

class LiteClassifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.block = nn.Sequential(
            _DSConv(in_channels, in_channels, stride=1),
            _DSConv(in_channels, in_channels, stride=1),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )

    def forward(self, x):
        return self.block(x)


# ================================================
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
        # avgpool = nn.AdaptiveAvgPool2d(size)
        # return avgpool(x) 
        return F.adaptive_avg_pool2d(x, output_size=size)

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


class FAscnn_pp_V17(nn.Module):
    def __init__(self, in_channels=3, num_classes=19, ffm_out=128):
        super().__init__()

        # Spatial / Learning-to-downsample
        self.spatial = LearningToDownsampleV2(
            in_channels=in_channels,
            c1=32,  
            c2=48,
            c3=64
        )  # -> (B,64,H/8,W/8)


        self.global_feature_extractor = GlobalFeatureExtractor(64, [64, 96, 128], 128, 6, [3, 3, 3])


        # BiSeNet-like FFM
        self.ffm = BiSeNetFFM(
            spatial_in=64,
            context_in=128,
            out_channels=ffm_out,
            reduction=4
        )  # -> (B,ffm_out,H/8,W/8)

        # Classifier
        self.classifier = LiteClassifier(ffm_out, num_classes)

        # Auxiliary head na context dla deep supervision
        self.aux_head = nn.Sequential(
            _DSConv(128, 128, stride=1),
            nn.Dropout(0.1),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )

    def forward(self, x):
        size = x.shape[2:]

        # spatial path
        spatial = self.spatial(x)  # H/8

        # context path
        context = self.global_feature_extractor(spatial)

        # do H/8 przed fusion
        context_up = F.interpolate(
            context,
            size=spatial.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        # fuse
        fused = self.ffm(spatial, context_up)

        # main head
        logits = self.classifier(fused)
        logits = F.interpolate(logits, size=size, mode='bilinear', align_corners=False)

        # if self.training:
        #     aux = self.aux_head(context)
        #     aux = F.interpolate(aux, size=size, mode='bilinear', align_corners=False)
        #     return {
        #         "main": logits,
        #         "aux": aux
        #     }

        return logits

    def __type__(self):
        return "FAscnn_pp_v17"


class BiSeNetFFM_Att(nn.Module):
    def __init__(self, spatial_in, context_in, out_channels, reduction=4, att_qk=32):
        super().__init__()
        fusion_in = spatial_in + context_in
        hidden = max(out_channels // reduction, 8)

        self.convblk = nn.Sequential(
            nn.Conv2d(fusion_in, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.fast_att = FastAttention(out_channels, att_qk)
        self.gamma = nn.Parameter(torch.zeros(1))

        # self.attention = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(out_channels, hidden, kernel_size=1, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(hidden, out_channels, kernel_size=1, bias=False),
        #     nn.Sigmoid(),
        # )

    def forward(self, spatial_feat, context_feat):
        feat = torch.cat([spatial_feat, context_feat], dim=1)
        feat = self.convblk(feat)

        feat = feat + self.gamma * self.fast_att(feat)
        # feat = feat + feat * (self.gamma * self.fast_att(feat))
        # att = self.attention(feat)
        # return feat + feat * att
        return feat

# fast attention na context. Reszta bez zmian.
class FAscnn_pp_V18(nn.Module):
    def __init__(self, in_channels=3, num_classes=19, ffm_out=128, ablation_cfg=None):
        super().__init__()
        
        self.ablation_cfg = ablation_cfg

        # Spatial / Learning-to-downsample
        self.spatial = LearningToDownsampleV2(
            in_channels=in_channels,
            c1=32,  
            c2=48,
            c3=64
        )  # -> (B,64,H/8,W/8)

        
        self.global_feature_extractor = GlobalFeatureExtractor(64, [64, 96, 128], 128, 6, [3, 3, 3])
        
        self.context_att = FastAttention(in_channels=128, embed_channels=32)
        self.context_gamma = nn.Parameter(torch.zeros(1))


        # BiSeNet-like FFM
        self.ffm = BiSeNetFFM(
            spatial_in=64,
            context_in=128,
            out_channels=ffm_out,
            reduction=4
        )  # -> (B,ffm_out,H/8,W/8)
        
        # self.ffm = BiSeNetFFM_Att(
        #     spatial_in=64,
        #     context_in=128,   
        #     out_channels=ffm_out,
        #     reduction=4
        # )  # -> (B,ffm_out,H/8,W/8)

        # Classifier
        self.classifier = LiteClassifier(ffm_out, num_classes)

        # Auxiliary head na context dla deep supervision
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
        size = x.shape[2:]

        # spatial path
        spatial = self.spatial(x)  # H/8

        # context path
        
        context = self.global_feature_extractor(spatial)
        
        use_attn = True
        if self.ablation_cfg is not None and not self.ablation_cfg.use_fa1:
            use_attn = False

        if use_attn:
            context = context + self.context_gamma * self.context_att(context)

        # do H/8 przed fusion
        context_up = F.interpolate(
            context,
            size=spatial.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        # fuse
        fused = self.ffm(spatial, context_up)

        # main head
        logits = self.classifier(fused)
        logits = F.interpolate(logits, size=size, mode='bilinear', align_corners=False)

        if self.training:
            aux_spatial = self.aux_downsaple(spatial)
            aux_spatial = F.interpolate(aux_spatial, size=size, mode='bilinear', align_corners=False)
            
            aux_context = self.aux_global(context)
            aux_context = F.interpolate(aux_context, size=size, mode='bilinear', align_corners=False)
                   
            return {
                "main": logits,
                "context": aux_context,
                "spatial": aux_spatial
            }

        return logits

    def __type__(self):
        return "FAscnn_pp_v18"


if __name__ == '__main__':
    import os
    import time
    import numpy as np
    from thop import profile

    def model_size_mb(model):
        return sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')  # wymuszam CPU do testów reparametryzacji

    if device.type == 'cpu':
        torch.set_num_threads(os.cpu_count())
        torch.set_num_interop_threads(1)

    img = torch.randn(1, 3, 1024, 2048).to(device)

    print(device)

    model = FAscnn_pp_V18(3, 19).to(device)
    model.eval()


    print("Model :", model.__type__())

    warmup_iters = 10
    timed_iters = 20

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
    p50_ms = np.percentile(times, 50)
    p95_ms = np.percentile(times, 95)
    p99_ms = np.percentile(times, 99)
    fps = 1000.0 / mean_ms

    print(f"Latency mean: {mean_ms:.3f} ms")
    print(f"Latency p50 : {p50_ms:.3f} ms")
    print(f"Latency p95 : {p95_ms:.3f} ms")
    print(f"Latency p99 : {p99_ms:.3f} ms")
    print(f"FPS         : {fps:.2f}")

    with torch.inference_mode():
        outputs = model(img)
    print(outputs.shape)

    flops, params = profile(model, inputs=(img,))
    print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
    print(f"Params: {params / 1e6:.2f} M")
    print(f"Model size: {model_size_mb(model):.2f} MB")
