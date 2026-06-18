from einops import repeat
from torch import nn
from einops.layers.torch import Rearrange
from torch import Tensor
import torch
import torch.nn.functional as F
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





if __name__ == "__main__":
    
    import time
    from thop import profile
    # img = torch.randn(1, 3, 256, 512)
    # img = torch.randn(1, 3, 360, 640)
    # img = torch.randn(1, 3, 720, 1280)
    # img = torch.randn(1, 3, 864,1600)
    
    img = torch.randn(1, 16, 512, 512)
    # img = torch.randn(1, 3, 1024, 1024)



    print('cpu')


    model = ViT(ch =16 , img_size=img.shape[2], patch_size=8, emb_dim=64,
                n_layers=6  , num_classes = 92, heads=4, dropout=0.1)
   


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

