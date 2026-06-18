from numpy import size
import torch
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import Tensor
import math


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


class LearningToDownsample(nn.Module):
    """Learning to downsample module"""

    def __init__(self, in_channels=32, dw_channels2=48, out_channels=64, **kwargs):
        super(LearningToDownsample, self).__init__()
        # self.conv = _ConvBNReLU(in_channels, dw_channels1, 3, 2)
        self.dsconv1 = _DSConv(in_channels, dw_channels2, 2)
        self.dsconv2 = _DSConv(dw_channels2, out_channels, 2)

    def forward(self, x):
        # x = self.conv(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        return x


class FeatureFusionModule(nn.Module):
    """Feature fusion module"""

    def __init__(self, highter_in_channels, lower_in_channels, out_channels, **kwargs):
        super(FeatureFusionModule, self).__init__()
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
        # size = lower_res_feature.size()[2:]
        size = higher_res_feature.size()[2:]
        lower_res_feature = self.dwconv(lower_res_feature)
        lower_res_feature = self.conv_lower_res(lower_res_feature)

        # higher_res_feature = F.interpolate(higher_res_feature, size=size, mode='bilinear', align_corners=True)
        lower_res_feature = F.interpolate(lower_res_feature, size=size, mode='bilinear', align_corners=True)
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
    
    
# Bottleneck Block - GroupNorm
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False, dilated=1, asymmetric=False, dropout_prob=0.05):
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



class AttnBlock(nn.Module):
    def __init__(self, channels: int, embed_channels: int):
        super().__init__()
        self.att = FastAttention(channels, embed_channels)
        self.norm = nn.BatchNorm2d(channels)   # lub GroupNorm(1, channels)
        self.proj = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x, gamma):
        y = self.att(x)
        y = self.proj(self.norm(y))
        return x + gamma * y

class FAscnn_pp_V13(nn.Module):
    def __init__(self,in_channels=3, num_classes = 19):
        super().__init__()
        
        self.gamma = nn.Parameter(torch.zeros(1))
        
        self.initial = InitialBlock(in_channels, out_channels=64)   
        # first branch (spatial path)
        
        self.downsample = LearningToDownsample(in_channels=64, dw_channels2=96, out_channels=128)
        
        # second branch (context path)
        # self.leyer1 = LinearBottleneck(128, 128, t=6, stride=1)
        self.leyer1 = nn.Sequential(
                Bottleneck(64, 128, downsample=True),
                Bottleneck(128, 128, dilated=2),
                Bottleneck(128, 128),
                Bottleneck(128, 128, downsample=True),
                Bottleneck(128, 128, dilated=4),
                Bottleneck(128, 128),  # usunięty asymmetric
            )
        
        
        self.leyer2 = nn.Sequential(    
                Bottleneck(128, 256, downsample=True),
                Bottleneck(256, 256, dilated=2),
                Bottleneck(256, 256),
                Bottleneck(256, 256, downsample=True),
                Bottleneck(256, 256, dilated=4),
                Bottleneck(256, 256),
            )
        self.att = AttnBlock(256, embed_channels=128)
        self.poll = PyramidPooling(256, 256)
        
        
        self.fusion = FeatureFusionModule(highter_in_channels=256, lower_in_channels=128, out_channels=128)
            
            
        self.final = Classifer(128, num_classes) 




    def forward(self, x):
        size = x.size()[2:]
        # innicial block   
        
        x = self.initial(x) 
        higher_res = self.downsample(x)

        lower_res = self.leyer1(x)
        lower_res = self.leyer2(lower_res)
        lower_res = self.att(lower_res, self.gamma)
        lower_res = self.poll(lower_res)


        x = self.fusion(lower_res, higher_res)
        
        x = self.final(x)
        x = F.interpolate(x, size, mode='bilinear', align_corners=False)
        return x
    
    def __type__(self):
        return "FAscnn_pp_v13"

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int, patch_size: int, emb_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, emb_size)
        )

    def forward(self, x):
        # x: (B, C, H, W) -> tokens: (B, N, E)
        B, C, H, W = x.shape
        p = self.patch_size
        assert H % p == 0 and W % p == 0, (H, W, p)
        Hp, Wp = H // p, W // p
        tokens = self.proj(x)  # (B, Hp*Wp, E)
        return tokens, Hp, Wp


class Unpatch(nn.Module):
    def __init__(self, emb_size: int, out_channels: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.to_patch = nn.Linear(emb_size, out_channels * patch_size * patch_size)

    def forward(self, tokens, Hp: int, Wp: int):
        # tokens: (B, N, E), N=Hp*Wp
        B, N, E = tokens.shape
        p = self.patch_size
        x = self.to_patch(tokens)  # (B, N, outC*p*p)
        x = x.view(B, Hp, Wp, -1)  # (B, Hp, Wp, outC*p*p)
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, outC*p*p, Hp, Wp)
        # "odpatchuj" do (B, outC, Hp*p, Wp*p)
        x = x.view(B, -1, Hp, Wp)               # (B, outC*p*p, Hp, Wp)
        x = F.pixel_shuffle(x, upscale_factor=p) # wymaga kanałów podzielnych przez p^2
        return x

class PatchEmbedConv(nn.Module):
    def __init__(self, in_channels: int, patch_size: int, emb_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels, emb_size,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False
        )

    def forward(self, x):
        # x: (B, C, H, W) -> (B, E, Hp, Wp)
        x = self.proj(x)
        B, E, Hp, Wp = x.shape
        tokens = x.flatten(2).transpose(1, 2).contiguous()  # (B, N, E), N=Hp*Wp
        return tokens, Hp, Wp

class UnpatchDeconv(nn.Module):
    def __init__(self, emb_size: int, out_channels: int, patch_size: int):
        super().__init__()
        self.out_channels = out_channels
        self.deproj = nn.ConvTranspose2d(
            emb_size, out_channels,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False
        )

    def forward(self, tokens, Hp: int, Wp: int):
        # tokens: (B, N, E)
        B, N, E = tokens.shape
        x = tokens.transpose(1, 2).contiguous().view(B, E, Hp, Wp)  # (B, E, Hp, Wp)
        x = self.deproj(x)  # (B, outC, Hp*p, Wp*p)
        return x

class FastAttentionTokens1D(nn.Module):
    def __init__(self, emb_dim: int, qk_dim: int):
        super().__init__()
        self.to_q = nn.Conv1d(emb_dim, qk_dim, 1, bias=False)
        self.to_k = nn.Conv1d(emb_dim, qk_dim, 1, bias=False)
        self.to_v = nn.Conv1d(emb_dim, emb_dim, 1, bias=False)

    def forward(self, x_tokens):
        # (B, N, E)
        B, N, E = x_tokens.shape
        x = x_tokens.transpose(1, 2).contiguous()  # (B, E, N)

        Q = F.normalize(self.to_q(x), dim=1)  # (B, qk, N)
        K = F.normalize(self.to_k(x), dim=1)  # (B, qk, N)
        V = self.to_v(x)                      # (B, E,  N)

        KV = torch.bmm(K, V.transpose(1, 2))              # (B, qk, E)
        Y  = torch.bmm(Q.transpose(1, 2), KV) / float(N)  # (B, N, E)
        return Y


class TokenAttnBlock(nn.Module):
    def __init__(self, emb_dim: int, qk_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(emb_dim)
        self.att = FastAttentionTokens1D(emb_dim, qk_dim)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x + self.gamma * self.att(self.norm(x))


class FAscnn_pp_V14(nn.Module):
    def __init__(self, in_channels=3, num_classes=19):
        super().__init__()

        self.initial = InitialBlock(in_channels, out_channels=64)
        self.downsample = LearningToDownsample(in_channels=64, dw_channels2=96, out_channels=128)

        self.leyer1 = nn.Sequential(
            Bottleneck(64, 128, downsample=True),
            Bottleneck(128, 128, dilated=2),
            Bottleneck(128, 128),
            Bottleneck(128, 128, downsample=True),
            Bottleneck(128, 128, dilated=4),
            Bottleneck(128, 128),
        )

        self.leyer2 = nn.Sequential(
            Bottleneck(128, 256, downsample=True),
            Bottleneck(256, 256, dilated=2),
            Bottleneck(256, 256),
            Bottleneck(256, 256, downsample=True),
            Bottleneck(256, 256, dilated=4),
            Bottleneck(256, 256),
        )

        self.poll = PyramidPooling(256, 256)

        # --- patch/token path na higher_res (B,128,H/8,W/8) ---
        patch_size = 8               # na MAPIE 1/8 (czyli tokeny z 8x8 komórek tej mapy)
        emb_dim = 128                # ustaw na 128, żeby było prosto
        qk_dim = 64                  # lżejsze Q/K

        self.emb = PatchEmbedConv(in_channels=128, patch_size=patch_size, emb_size=emb_dim)
        self.token_att = TokenAttnBlock(emb_dim=emb_dim, qk_dim=qk_dim)
        self.unpatch = Unpatch(emb_size=emb_dim, out_channels=128, patch_size=patch_size)

        self.fusion = FeatureFusionModule(highter_in_channels=256, lower_in_channels=128, out_channels=128)
        self.final = Classifer(128, num_classes)

    def forward(self, x):
        size = x.size()[2:]

        x0 = self.initial(x)              # (B,64,H/2,W/2)
        spatial = self.downsample(x0)     # (B,128,H/8,W/8)

        context = self.leyer1(x0)
        context = self.leyer2(context)    # (B,256,H/32,W/32)
        context = self.poll(context)      # (B,256,H/32,W/32)

        # patch + token attention + unpatch na ścieżce spatial
        tokens, Hp, Wp = self.emb(spatial)     # (B, Hp*Wp, emb_dim)
        tokens = self.token_att(tokens)        # (B, Hp*Wp, emb_dim)
        spatial2 = self.unpatch(tokens, Hp, Wp) # (B,128,H/8,W/8)

        fused = self.fusion(context, spatial2)  # -> (B,128,H/8,W/8) bo fusion upsample'uje low->high
        out = self.final(fused)
        out = F.interpolate(out, size, mode='bilinear', align_corners=False)
        return out

    def __type__(self):
        return "FAscnn_pp_v14"

class FFNBlock(nn.Module):
    """
    Klasyczny FFN z ViT (PreNorm), do wpięcia *po attention*.
    Koszt ~ O(N * D * mlp_ratio), bez O(N^2).
    """
    def __init__(self, emb_dim: int, mlp_ratio: int = 3, drop: float = 0.0, gamma_init: float = 1e-3):
        super().__init__()
        hidden = int(emb_dim * mlp_ratio)
        self.ln = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, emb_dim),
            nn.Dropout(drop),
        )
        self.gamma = nn.Parameter(torch.ones(emb_dim) * gamma_init)
        
    def forward(self, x):
        # x: (B, N, D)
        x = x + self.gamma * self.mlp(self.ln(x))
        return x

class P2RefineHead(nn.Module):
    """
    P2 (1/4) refinement head:
    - bierze C2 z x0 (H/2 -> H/4) i P2 z fused (H/8 -> H/4)
    - bardzo lekki smoothing + predykcja klas
    """

    def __init__(self, c2_in=16, p_in=64, p2_channels=32, num_classes=19):
        super().__init__()
        self.c2_down = nn.AvgPool2d(kernel_size=2, stride=2)  # H/2 -> H/4

        self.c2_lat = nn.Sequential(
            nn.Conv2d(c2_in, p2_channels, 1, bias=False),
            nn.BatchNorm2d(p2_channels),
        )
        self.p_lat = nn.Sequential(
            nn.Conv2d(p_in, p2_channels, 1, bias=False),
            nn.BatchNorm2d(p2_channels),
        )

        # ultra-light smoothing: DW 3x3 + PW 1x1
        self.dw = nn.Conv2d(p2_channels, p2_channels, 3, padding=1, groups=p2_channels, bias=False)
        self.pw = nn.Sequential(
            nn.Conv2d(p2_channels, p2_channels, 1, bias=False),
            nn.BatchNorm2d(p2_channels),
            nn.ReLU(inplace=True),
        )

        # predykcja: bez 2x DSConv (oszczędzasz sporo na 1/4)
        self.cls = nn.Conv2d(p2_channels, num_classes, 1, bias=True)

    def forward(self, x0_h2, fused_h8):
        # C2: x0 (H/2) -> (H/4)
        c2 = self.c2_down(x0_h2)
        c2 = self.c2_lat(c2)

        # P2: fused (H/8) -> (H/4)
        p2 = F.interpolate(fused_h8, scale_factor=2, mode="bilinear", align_corners=False)
        p2 = self.p_lat(p2)

        y = p2 + c2
        y = self.pw(self.dw(y))
        logits_h4 = self.cls(y)
        return logits_h4
    
    
class FAscnn_pp_V15(nn.Module):
    def __init__(self, in_channels=3, num_classes=19):
        super().__init__()

        self.initial = InitialBlock(in_channels, out_channels=16)
        self.downsample = LearningToDownsample(in_channels=16, dw_channels2=96, out_channels=128)

        self.leyer1 = nn.Sequential(
            Bottleneck(16, 32, downsample=True),
            Bottleneck(32, 32, dilated=2),
            Bottleneck(32, 32),
            Bottleneck(32, 64, downsample=True),
            Bottleneck(64, 64, dilated=4),
            Bottleneck(64, 64),
        )
        
        # --- patch/token path na higher_res (B,128,H/8,W/8) ---
        patch_size = 4               # na MAPIE 1/8 (czyli tokeny z 8x8 komórek tej mapy)
        emb_dim = 128                # ustaw na 128, żeby było prosto
        qk_dim = 64                  # lżejsze Q/K

        
        self.emb1 = PatchEmbedConv(in_channels=128, patch_size=patch_size, emb_size=emb_dim)
        self.token_att1 = TokenAttnBlock(emb_dim=emb_dim, qk_dim=qk_dim)
        self.ffn1 = FFNBlock(emb_dim=emb_dim, mlp_ratio=3, drop=0.0)
        self.unpatch1 = Unpatch(emb_size=emb_dim, out_channels=128, patch_size=patch_size)
        
        self.emb2 = PatchEmbedConv(in_channels=128, patch_size=patch_size, emb_size=emb_dim)
        self.token_att2 = TokenAttnBlock(emb_dim=emb_dim, qk_dim=qk_dim)
        self.ffn2 = FFNBlock(emb_dim=emb_dim, mlp_ratio=3, drop=0.0)
        self.unpatch2 = Unpatch(emb_size=emb_dim, out_channels=128, patch_size=patch_size)

        self.fusion = FeatureFusionModule(highter_in_channels=128, lower_in_channels=64, out_channels=64)
        self.final = Classifer(64, num_classes)
        
        self.p2_head = P2RefineHead(
            c2_in=16,
            p_in=64,
            p2_channels=32,   # 32 = bezpieczne dla FPS
            num_classes=num_classes
        )

    def forward(self, x):
        size = x.size()[2:]

        x0 = self.initial(x)              # (B,64,H/2,W/2)
        spatial = self.downsample(x0)     # (B,128,H/8,W/8)

        detail = self.leyer1(x0) # (B,64,H/8,W/8)


        # patch + token attention + unpatch na ścieżce spatial
        tokens, Hp, Wp = self.emb1(spatial)     # (B, Hp*Wp, emb_dim)
        tokens = self.token_att1(tokens)        # (B, Hp*Wp, emb_dim)
        tokens = self.ffn1(tokens)               # (B, Hp*Wp, emb_dim)
        spatial = self.unpatch1(tokens, Hp, Wp) # (B,128,H/8,W/8)

        tokens, Hp, Wp = self.emb2(spatial)     # (B, Hp*Wp, emb_dim)  
        tokens = self.token_att2(tokens)        # (B, Hp*Wp, emb_dim)
        tokens = self.ffn2(tokens)               # (B, Hp*Wp, emb_dim)
        spatial = self.unpatch2(tokens, Hp, Wp) # (B,128,H/8,W/8)

        fused = self.fusion(spatial, detail)  # -> (B,128,H/8,W/8) bo fusion upsample'uje low->high
        # out = self.final(fused)
        out = self.p2_head(x0, fused)  # (B, num_classes, H/4, W/4)
        out = F.interpolate(out, size, mode='bilinear', align_corners=False)
        return out

    def __type__(self):
        return "FAscnn_pp_v15"


class DWConvBNReLU(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class BetterFusionModule(nn.Module):
    """
    Lepsza fuzja niż samo 1x1 + suma.
    - wyrównanie kanałów
    - prosty gate
    - lokalne doszlifowanie 3x3 DW + PW
    """
    def __init__(self, high_in_channels, low_in_channels, out_channels):
        super().__init__()

        self.high_proj = nn.Sequential(
            nn.Conv2d(high_in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.low_proj = nn.Sequential(
            nn.Conv2d(low_in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.gate = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 1, bias=True),
            nn.Sigmoid()
        )

        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, high_res_feature, low_res_feature):
        # high_res_feature: wyższa rozdzielczość
        # low_res_feature: niższa rozdzielczość
        size = high_res_feature.shape[2:]

        low = self.low_proj(low_res_feature)
        low = F.interpolate(low, size=size, mode='bilinear', align_corners=False)

        high = self.high_proj(high_res_feature)

        gate = self.gate(torch.cat([high, low], dim=1))
        fused = high + gate * low

        return self.refine(fused)


class SegAuxHead(nn.Module):
    """
    Lekki auxiliary head do deep supervision.
    """
    def __init__(self, in_channels, mid_channels, num_classes):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(mid_channels, num_classes, 1)
        )

    def forward(self, x, out_size=None):
        x = self.block(x)
        if out_size is not None:
            x = F.interpolate(x, out_size, mode='bilinear', align_corners=False)
        return x


class BoundaryHead(nn.Module):
    """
    1-kanałowa mapa granicy.
    """
    def __init__(self, in_channels, mid_channels=32):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, 1)
        )

    def forward(self, x, out_size=None):
        x = self.block(x)
        if out_size is not None:
            x = F.interpolate(x, out_size, mode='bilinear', align_corners=False)
        return x


class BetterP2RefineHead(nn.Module):
    """
    Mocniejsza wersja Twojego P2RefineHead.
    - bierze x0 z H/2
    - bierze fused z H/8
    - łączy na H/4
    - dwa lokalne bloki wygładzające
    """
    def __init__(self, c2_in=24, p_in=96, p2_channels=64, num_classes=19):
        super().__init__()

        self.c2_down = nn.AvgPool2d(kernel_size=2, stride=2)  # H/2 -> H/4

        self.c2_lat = nn.Sequential(
            nn.Conv2d(c2_in, p2_channels, 1, bias=False),
            nn.BatchNorm2d(p2_channels),
        )
        self.p_lat = nn.Sequential(
            nn.Conv2d(p_in, p2_channels, 1, bias=False),
            nn.BatchNorm2d(p2_channels),
        )

        self.refine1 = nn.Sequential(
            nn.Conv2d(p2_channels, p2_channels, 3, padding=1, groups=p2_channels, bias=False),
            nn.BatchNorm2d(p2_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(p2_channels, p2_channels, 1, bias=False),
            nn.BatchNorm2d(p2_channels),
            nn.ReLU(inplace=True),
        )

        self.refine2 = nn.Sequential(
            nn.Conv2d(p2_channels, p2_channels, 3, padding=1, groups=p2_channels, bias=False),
            nn.BatchNorm2d(p2_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(p2_channels, p2_channels, 1, bias=False),
            nn.BatchNorm2d(p2_channels),
            nn.ReLU(inplace=True),
        )

        self.cls = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(p2_channels, num_classes, 1)
        )

    def forward(self, x0_h2, fused_h8):
        c2 = self.c2_down(x0_h2)  # H/2 -> H/4
        c2 = self.c2_lat(c2)

        p2 = F.interpolate(fused_h8, scale_factor=2, mode='bilinear', align_corners=False)  # H/8 -> H/4
        p2 = self.p_lat(p2)

        y = c2 + p2
        y = self.refine1(y)
        y = self.refine2(y)
        logits_h4 = self.cls(y)
        return logits_h4


class FAscnn_pp_V16(nn.Module):
    def __init__(self, in_channels=3, num_classes=19):
        super().__init__()

        # trochę szerszy shallow stem niż w V15
        self.initial = InitialBlock(in_channels, out_channels=24)      # H/2
        self.downsample = LearningToDownsample(
            in_channels=24, dw_channels2=96, out_channels=128
        )  # H/8

        # detail branch: trochę szersza i nadal lekka
        self.leyer1 = nn.Sequential(
            Bottleneck(24, 48, downsample=True),   # H/4
            Bottleneck(48, 48, dilated=2),
            Bottleneck(48, 48),
            Bottleneck(48, 96, downsample=True),   # H/8
            Bottleneck(96, 96, dilated=4),
            Bottleneck(96, 96),
        )

        # tylko jeden blok tokenowy
        patch_size = 4
        emb_dim = 128
        qk_dim = 64

        self.emb = PatchEmbedConv(in_channels=128, patch_size=patch_size, emb_size=emb_dim)
        self.token_att = TokenAttnBlock(emb_dim=emb_dim, qk_dim=qk_dim)
        self.ffn = FFNBlock(emb_dim=emb_dim, mlp_ratio=3, drop=0.0)
        self.unpatch = UnpatchDeconv(emb_size=emb_dim, out_channels=128, patch_size=patch_size)

        # lepsza fuzja spatial/detail
        self.fusion = BetterFusionModule(
            high_in_channels=128,   # spatial
            low_in_channels=96,     # detail
            out_channels=96
        )

        # główna głowica
        self.p2_head = BetterP2RefineHead(
            c2_in=24,
            p_in=96,
            p2_channels=64,
            num_classes=num_classes
        )

        # auxiliary heads
        self.aux_detail = SegAuxHead(in_channels=96, mid_channels=64, num_classes=num_classes)
        self.aux_spatial = SegAuxHead(in_channels=128, mid_channels=64, num_classes=num_classes)

        # boundary head
        self.boundary_head = BoundaryHead(in_channels=96, mid_channels=32)

    def forward(self, x):
        size = x.shape[2:]

        x0 = self.initial(x)            # (B,24,H/2,W/2)
        spatial = self.downsample(x0)   # (B,128,H/8,W/8)

        detail = self.leyer1(x0)        # (B,96,H/8,W/8)

        # jeden token block
        tokens, Hp, Wp = self.emb(spatial)     # (B, N, 128)
        tokens = self.token_att(tokens)
        tokens = self.ffn(tokens)
        spatial = self.unpatch(tokens, Hp, Wp) # (B,128,H/8,W/8)

        fused = self.fusion(spatial, detail)   # (B,96,H/8,W/8)

        main = self.p2_head(x0, fused)         # (B,num_classes,H/4,W/4)
        main = F.interpolate(main, size, mode='bilinear', align_corners=False)

        # if self.training:
        #     aux_detail = self.aux_detail(detail, out_size=size)
        #     aux_spatial = self.aux_spatial(spatial, out_size=size)
        #     boundary = self.boundary_head(fused, out_size=size)
        #     return {
        #         "main": main,
        #         "aux_detail": aux_detail,
        #         "aux_spatial": aux_spatial,
        #         "boundary": boundary
        #     }

        return main

    def __type__(self):
        return "FAscnn_pp_v16"



if __name__ == '__main__':

    import time
    from thop import profile


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    
    if device.type == 'cpu':
        torch.set_num_threads(os.cpu_count())
        torch.set_num_interop_threads(1)
    
    # img = torch.randn(1, 3, 256, 512).to(device)
    # img = torch.randn(1, 3, 360, 640).to(device)
    # img = torch.randn(1, 3, 720, 1280)
    img = torch.randn(1, 3, 1024,2048).to(device)
    
    print(device)
    

    model = FAscnn_pp_V16(3,19).to(device)
    model.eval()
    
    print("Model : ", model.__type__())
    
    for _ in range(10):
        _ = model(img)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()
    outputs = model(img)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end = time.perf_counter()

    print(outputs.shape)



    elapsed_ms = (end - start) * 1000
    print(f"Czas wykonania: {elapsed_ms:.3f} ms")
    fps = 1 / (end - start)
    print(f"FPS: {fps:.2f}")


    flops, params = profile(model, inputs=(img,))
    print(f"FLOPs: {flops/1e9:.2f} GFLOPs")
    print(f"Params: {params/1e6:.2f} M")

