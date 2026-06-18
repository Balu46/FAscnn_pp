import torch
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import Tensor
import math






def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    # (batch, heads, seq_len, head_dim) @ (batch, heads, head_dim, seq_len) --> (batch, heads, seq_len, seq_len)
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
    if mask is not None:
        scaled += mask
    attention = F.softmax(scaled, dim=-1)
    # (batch, heads, seq_len, seq_len) @ (batch, heads, seq_len, head_dim) --> (batch, heads, seq_len, head_dim)
    values = torch.matmul(attention, v)
    return values, attention

class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, d_model, num_heads):
        super().__init__()
        self.input_dim = input_dim      # Input embedding size
        self.d_model = d_model          # Model embedding size (output of self-attention)
        self.num_heads = num_heads      # Number of parallel attention heads
        self.head_dim = d_model // num_heads  # Dimensionality per head

        # For efficiency, compute Q, K, V for all heads at once with a single linear layer
        self.qkv_layer = nn.Linear(input_dim, 3 * d_model)
        # Final projection, combines all heads' outputs
        self.linear_layer = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, sequence_length, input_dim = x.size()
        # print(f"x.size(): {x.size()}")  # Input shape

        # Step 1: Project x into concatenated q, k, v for ALL heads at once
        qkv = self.qkv_layer(x)
        # print(f"qkv.size(): {qkv.size()}")  # Shape: (batch, seq_len, 3 * d_model)

        # Step 2: reshape into (batch, seq_len, num_heads, 3 * head_dim)
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim)
        # print(f"qkv.size(): {qkv.size()}")

        # Step 3: Rearrange to (batch, num_heads, seq_len, 3 * head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        # print(f"qkv.size(): {qkv.size()}")

        # Step 4: Split the last dimension into q, k, v (each get last dimension of head_dim)
        q, k, v = qkv.chunk(3, dim=-1)  # Each: (batch, num_heads, seq_len, head_dim)
        # print(f"q size: {q.size()}, k size: {k.size()}, v size: {v.size()}")

        # Step 5: Apply scaled dot product attention to get outputs (contextualized values) and attention weights
        values, attention = scaled_dot_product(q, k, v, mask)
        # print(f"values.size(): {values.size()}, attention.size: {attention.size()}")

        # Step 6: Merge the heads (permute before reshape)
        values = values.permute(0, 2, 1, 3)   # (batch, seq_len, heads, head_dim)
        values = values.reshape(batch_size, sequence_length, self.num_heads * self.head_dim)
        # print(f"values.size(): {values.size()}")

        # Step 7: Final linear projection to match d_model
        out = self.linear_layer(values)
        # print(f"out.size(): {out.size()}")
        
        return out


class FeedForward(nn.Sequential):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)            
        )
        
class Attention(nn.Module):
    def __init__(self, dim, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.att = MultiheadAttention(input_dim=dim, d_model=dim, num_heads=n_heads)
        # self.q = torch.nn.Linear(dim, dim)
        # self.k = torch.nn.Linear(dim, dim)
        # self.v = torch.nn.Linear(dim, dim)

    def forward(self, x):
        # q = self.q(x)
        # k = self.k(x)
        # v = self.v(x)
        attn_output  = self.att(x)
        return attn_output

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
    
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels = 3, patch_size = 8, emb_size = 128):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # break-down the image in s1 x s2 patches and flat them
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, emb_size)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x)
        return x


class ViT(nn.Module):
    def __init__(
        self,
        ch=3,
        img_size=144,
        patch_size=4,
        emb_dim=64,
        n_layers=6,
        num_classes=19,
        heads=4,
        dropout=0.1
    ):
        super().__init__()

        self.patch_size = patch_size

        self.patch_embedding = PatchEmbedding(
            in_channels=ch,
            patch_size=patch_size,
            emb_size=emb_dim
        )

        num_patches = (img_size // patch_size) ** 2

        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches, emb_dim)
        )

        self.layers = nn.ModuleList([
            nn.Sequential(
                ResidualAdd(
                    PreNorm(emb_dim, Attention(emb_dim, heads, dropout))
                ),
                ResidualAdd(
                    PreNorm(
                        emb_dim,
                        FeedForward(emb_dim, emb_dim * 4, dropout)
                    )
                )
            )
            for _ in range(n_layers)
        ])

        self.seg_head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, num_classes)
        )

    def forward(self, img):
        x = self.patch_embedding(img)
        b, n, _ = x.shape

        x = x + self.pos_embedding[:, :n]

        for layer in self.layers:
            x = layer(x)

        x = self.seg_head(x)          # (B, N, C)

        h = w = int(math.sqrt(n))
        x = x.permute(0, 2, 1)        # (B, C, N)
        x = x.view(b, -1, h, w)       # (B, C, H_p, W_p)

        x = F.interpolate(
            x,
            scale_factor=self.patch_size,
            mode="bilinear",
            align_corners=False
        )

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

    def forward(self, x):
        x =  self.conv(x)
        return self.normalize(x)
# Bottleneck Block - GroupNorm
class Bottleneck(nn.Module):
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


class FeatureFusionModule(nn.Module):
    """Feature fusion module"""

    def __init__(self, highter_in_channels, lower_in_channels, out_channels, scale_factor=4, **kwargs):
        super(FeatureFusionModule, self).__init__()
        # self.scale_factor = scale_factor
        self.scale_factor = highter_in_channels / lower_in_channels
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
        # lower_res_feature = F.interpolate(lower_res_feature, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
        lower_res_feature = self.dwconv(lower_res_feature)
        lower_res_feature = self.conv_lower_res(lower_res_feature)

        higher_res_feature = self.conv_higher_res(higher_res_feature)
        out = higher_res_feature + lower_res_feature
        return self.relu(out)

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




class FAscnn_pp_V11(nn.Module):
    def __init__(self,in_channels=3, num_classes = 19):
        super().__init__()
        self.initial = InitialBlock(in_channels)

        self.bottleneck1 = nn.Sequential(
            Bottleneck(16, 64, downsample=True),
            Bottleneck(64, 64),
            Bottleneck(64, 128, downsample=True),
            # Bottleneck(128, 128),
            Bottleneck(128, 128, dilated=2),
            Bottleneck(128, 128),
            Bottleneck(128, 128, dilated=4)
        )


        self.att_1 = FastAttention(in_channels=16, embed_channels=64)
        self.att_2 = FastAttention(in_channels=64, embed_channels=64)

        self.bottleneck2 = nn.Sequential(
            Bottleneck(16, 64, downsample=True),
            Bottleneck(64, 64),
            Bottleneck(64, 64, dilated=2),
            # Bottleneck(128, 128),  # usunięty asymmetric
            # Bottleneck(128, 128, dilated=4),
        )

        self.vit = ViT(
            ch=16,
            img_size=256,
            patch_size=4,
            emb_dim=64,
            n_layers=6,
            num_classes=64)
        
        # Dekoder z Upsample bilinear + Conv2d

        # self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        # self.up2 = nn.ConvTranspose2d(64, 16, kernel_size=2, stride=2)
        # self.final = nn.ConvTranspose2d(16, num_classes, kernel_size=2, stride=2)
        
        self.up = FeatureFusionModule(64, 128, 128, scale_factor=2)
        self.final = Classifer(128, num_classes)    



    def forward(self, x):
        size = x.size()[2:]  # HxW
        # innicial block
        x = self.initial(x)
        
        # first branch
        x_res = torch.matmul(self.att_1(x), x)   # 16 channels
        # print( x_res.shape)
        x = self.bottleneck1(x)
        
        # second branch
        # x_res = self.vit(x_res)
        # x_res = F.interpolate(x_res, scale_factor=0.5, mode='bilinear', align_corners=True)
        
        x_res = self.bottleneck2(x_res)
        
        x_res = torch.matmul(self.att_2(x_res), x_res) # 64 channels
        # print( x_res.shape)
        
        
        # combine branches
        x = self.up( x_res, x)
        
        x = self.final(x)
        
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        
        
        return x
    
    def __type__(self):
        return "FAscnn_pp_v11"


class Multihead_Fast_Attention(nn.Module):
    def __init__(self, in_channels: int, embed_channels: int, num_heads: int):
        """
        in_channels: liczba kanałów wejściowych C
        embed_channels: liczba kanałów dla Q i K (C')
        num_heads: liczba głów
        """
        super().__init__()

        assert embed_channels % num_heads == 0, \
            "embed_channels musi być podzielne przez num_heads"

        self.in_channels = in_channels
        self.embed_channels = embed_channels
        self.num_heads = num_heads
        self.head_dim = embed_channels // num_heads

        self.to_q = nn.Conv2d(in_channels, embed_channels, kernel_size=1, bias=False)
        self.to_k = nn.Conv2d(in_channels, embed_channels, kernel_size=1, bias=False)
        self.to_v = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)

        # opcjonalna projekcja wyjścia (często pomaga)
        self.proj = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        n = H * W
        h = self.num_heads

        # --- Q, K, V ---
        Q = self.to_q(x)   # (B, C', H, W)
        K = self.to_k(x)   # (B, C', H, W)
        V = self.to_v(x)   # (B, C,  H, W)

        # --- reshape do multi-head ---
        # Q, K: (B, h, head_dim, n)
        Q = Q.view(B, h, self.head_dim, n)
        K = K.view(B, h, self.head_dim, n)

        # V: (B, h, C/h, n)
        V = V.view(B, h, C // h, n)

        # --- normalizacja ---
        Q = F.normalize(Q, dim=2)
        K = F.normalize(K, dim=2)

        # --- Fast Attention ---
        # K^T @ V : (B, h, head_dim, C/h)
        KV = torch.einsum('bhcn,bhun->bhcu', K, V)

        # Q @ (K^T V) : (B, h, n, C/h)
        Y = torch.einsum('bhcn,bhcu->bhnu', Q, KV)

        Y = Y / n

        # --- merge głów ---
        Y = Y.permute(0, 1, 3, 2).contiguous()   # (B, h, C/h, n)
        Y = Y.view(B, C, H, W)

        return self.proj(Y)




class FAscnn_pp_V12(nn.Module):
    def __init__(self,in_channels=3, num_classes = 19, num_layers=1, num_heads=4):
        super().__init__()
        self.initial = InitialBlock(in_channels, out_channels=64)
        self.num_layers = num_layers    
        self.layers = nn.ModuleList()
        
        for _ in range(num_layers):
            
            self.layers.append(nn.ModuleDict()) 
            if _ != 0 :
                self.layers[-1]['upp'] = nn.ConvTranspose2d(
                    in_channels=128,
                    out_channels=64,
                    kernel_size=2,
                    stride=2
                )
            self.layers[-1]['att'] = Multihead_Fast_Attention(in_channels=64, embed_channels=128, num_heads=num_heads)
            self.layers[-1]['bottleneck1'] = nn.Sequential(
                Bottleneck(64, 128, downsample=True),
                Bottleneck(128, 128, dilated=2),
                Bottleneck(128, 128),
                # Bottleneck(128, 128),  # usunięty asymmetric
                # Bottleneck(128, 128, dilated=4),
            )
            self.layers[-1]['deconv'] = nn.ConvTranspose2d(
                in_channels=128,
                out_channels=128,
                kernel_size=2,
                stride=2
            )

            self.layers[-1]['fuse1'] = FeatureFusionModule(64, 64, 64, scale_factor=2)

            self.layers[-1]['fuse2'] = FeatureFusionModule(64, 128, 128, scale_factor=2)

        
        
        if self.num_layers == 1:
            self.up = nn.ConvTranspose2d(
                    in_channels=128,
                    out_channels=128,
                    kernel_size=2,
                    stride=2
                )
        
        
        
        
        self.final = Classifer(128, num_classes) 
       



    def forward(self, x):
        # innicial block   
        
        x = self.initial(x) 


        for layer in range(len(self.layers)):
            if 'upp' in self.layers[layer]:
                x = self.layers[layer]['upp'](x)
            x_res = x
            x = self.layers[layer]['att'](x)
            x = self.layers[layer]['fuse1'](x_res, x)
            
            x_res = x
            x = self.layers[layer]['bottleneck1'](x)
            x = self.layers[layer]['deconv'](x)
            x = self.layers[layer]['fuse2'](x_res, x)


        if self.num_layers == 1:
            x = self.up(x)
            
        x = self.final(x)
        
        return x
    
    def __type__(self):
        return "FAscnn_pp_v12"







if __name__ == '__main__':

    import time
    from thop import profile

    # img = torch.randn(1, 3, 256, 512)
    # img = torch.randn(1, 3, 360, 640)
    # img = torch.randn(1, 3, 720, 1280)
    img = torch.randn(1, 3, 1024,2048)
    
    # img = torch.randn(1, 3, 512, 512)


    print('cpu')

    model = FAscnn_pp_V12(3,1, num_layers=1, num_heads=4)
    
    

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

