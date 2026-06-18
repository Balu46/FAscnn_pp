import torch
import torch.nn as nn
import torch.nn.functional as F


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


class model_patched(nn.Module):
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
            patch_out = self.model(patches)
            _, C_out, th, tw = patch_out.shape

            output = merge_patches(
                patch_out,
                coords,
                (B, C_out, H, W),
                (th, tw)
            )

            return output