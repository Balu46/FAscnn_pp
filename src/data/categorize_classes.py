import torch
import numpy as np
import argparse
from tqdm import tqdm
from torchvision.datasets.cityscapes import Cityscapes as CS
from torch.utils.data import DataLoader
import torchvision.transforms as T

def scan_dataset(data_loader, max_id=40, ignore_index=255):
    print(f"Rozpoczynam profilowanie zbioru pod kątem Copy-Paste na {data_loader.num_workers} wątkach...")
    total_pixels = 0
    class_pixels = torch.zeros(max_id, dtype=torch.float64)
    class_image_presence = torch.zeros(max_id, dtype=torch.float64)
    
    for batch in tqdm(data_loader, desc="Skanowanie masek"):
        _, masks = batch
        for i in range(masks.shape[0]):
            mask = masks[i]
            valid_mask = mask[mask != ignore_index]
            total_pixels += valid_mask.numel()
            counts = torch.bincount(valid_mask.flatten(), minlength=max_id)
            class_pixels += counts.double()
            class_image_presence += (counts > 0).double()
            
    return class_pixels.numpy(), class_image_presence.numpy(), total_pixels

def compute_taxonomy(class_pixels, class_image_presence, total_pixels, pixels_per_img, args):
    p_global = (class_pixels / total_pixels) * 100.0
    safe_presence = np.clip(class_image_presence, a_min=1.0, a_max=None)
    area_cond = ((class_pixels / safe_presence) / pixels_per_img) * 100.0
    
    paste_classes, protect_classes = [], []
    id_to_name = {c.id: c.name for c in CS.classes if c.id >= 0}
    
    print("\n--- RAPORT TAKSONOMICZNY (Data-Driven Copy-Paste) ---")
    for c_id in range(len(class_pixels)):
        if class_pixels[c_id] == 0 or c_id not in id_to_name: continue
        target_train_id = CS.classes[c_id].train_id
        if target_train_id < 0 or target_train_id == 255: continue
            
        name = id_to_name[c_id]
        p_glob_pct, a_cond_pct = p_global[c_id], area_cond[c_id]
        
        if a_cond_pct > args.protect_area and p_glob_pct < args.protect_glob:
            protect_classes.append(c_id)
            print(f"[PROTECT] ID: {c_id:02d} ({name:15s}) | P_glob: {p_glob_pct:5.2f}%")
        elif p_glob_pct < args.paste_glob and a_cond_pct < args.paste_area:
            paste_classes.append(c_id)
            print(f"[PASTE]   ID: {c_id:02d} ({name:15s}) | P_glob: {p_glob_pct:5.2f}%")
            
    print("\n---> Do Datasetu (Dataloader):")
    print(f"self.rare_class_ids = np.array({paste_classes})")
    print(f"self.protected_class_ids = np.array([])  # R&D Decyzja: Zostawiamy puste dla OHEM")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--city_root', type=str, default="cityscapes")
    parser.add_argument('--protect_area', type=float, default=1.7)
    parser.add_argument('--protect_glob', type=float, default=0.7)
    parser.add_argument('--paste_area', type=float, default=1.0)
    parser.add_argument('--paste_glob', type=float, default=1.0)
    args = parser.parse_args()
    
    dataset = CS(root=args.city_root, split="train", mode="fine", target_type="semantic",
                 transform=T.ToTensor(), target_transform=lambda mask: torch.from_numpy(np.array(mask, dtype=np.int64)).long())
    loader = DataLoader(dataset, batch_size=8, num_workers=8)
    
    cp, cip, tp = scan_dataset(loader)
    compute_taxonomy(cp, cip, tp, 1024 * 2048, args)
    
        