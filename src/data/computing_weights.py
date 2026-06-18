import torch
import numpy as np
from tqdm import tqdm
from torchvision.datasets.cityscapes import Cityscapes as CS
from torch.utils.data import DataLoader
import torchvision.transforms as T

def scan_dataset(data_loader, max_id=40, ignore_index=255):
    print(f"Skanowanie 3D przestrzeni statystycznej zbioru...")
    total_pixels = 0
    total_images = 0
    class_pixels = torch.zeros(max_id, dtype=torch.float64)
    class_image_presence = torch.zeros(max_id, dtype=torch.float64)
    
    for batch in tqdm(data_loader, desc="Profilowanie sygnatur klas"):
        _, masks = batch
        total_images += masks.shape[0]
        for i in range(masks.shape[0]):
            mask = masks[i]
            valid_mask = mask[mask != ignore_index]
            total_pixels += valid_mask.numel()
            counts = torch.bincount(valid_mask.flatten(), minlength=max_id)
            class_pixels += counts.double()
            class_image_presence += (counts > 0).double()
            
    return class_pixels.numpy(), class_image_presence.numpy(), total_pixels, total_images

def generate_intelligent_weights(class_pixels, class_image_presence, total_pixels, total_images, pixels_per_img):
    # 1. ORYGINALNA BAZA DEEPLAB (Nienaruszalny fundament)
    base_weights = np.array([
        0.8373, 0.9180, 0.8660, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489, 
        0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955, 
        1.0865, 1.1529, 1.0507
    ])

    # 2. OBLICZENIA SYGNATUR (3 Wymiary)
    p_global = (class_pixels / total_pixels) * 100.0
    presence_rate = (class_image_presence / total_images) * 100.0
    safe_presence = np.clip(class_image_presence, a_min=1.0, a_max=None)
    area_cond = ((class_pixels / safe_presence) / pixels_per_img) * 100.0

    id_to_trainid = {c.id: c.train_id for c in CS.classes if c.id >= 0}
    multipliers = np.ones(19, dtype=np.float64)

    print("\n=================================================================")
    print(" ROZPOZNAWANIE OBIEKTÓW PRZEZ SYGNATURY STATYSTYCZNE")
    print("=================================================================")

    # 3. SILNIK DECYZYJNY (Maszyna rozumie zbiór)
    for c_id in range(len(class_pixels)):
        if c_id not in id_to_trainid: continue
        t_id = id_to_trainid[c_id]
        if t_id < 0 or t_id >= 19: continue
            
        p_glob_pct = p_global[c_id]
        pres_pct = presence_rate[c_id]
        area_pct = area_cond[c_id]
        name = CS.classes[c_id].name
        
        # FILTR 1: Obiekty pospolite i tło (Wypycha Drogę, Budynki, Naturę, Auta i Pieszych)
        if p_glob_pct > 0.8:
            print(f"[{name:15s}] Tło / Pospolite  | P_glob: {p_glob_pct:4.1f}% -> Boost: x1.0")
            continue
            
        # FILTR 2: Giganty (Wypycha Autobusy, Pociągi, Ciężarówki - bo wywołują halucynacje)
        if area_pct > 2.6:
            print(f"[{name:15s}] Rzadki Gigant    | Area: {area_pct:4.1f}%   -> Boost: x1.0")
            continue
            
        # FILTR 3: Cienka Architektura (Pojawiają się ciągle, ale są cieniutkie jak Słupy/Znaki)
        if pres_pct > 20.0:
            multipliers[t_id] = 1.5
            print(f"[{name:15s}] Cienka Infra     | Pres: {pres_pct:4.1f}%   -> Boost: x1.5")
            continue
            
        # FILTR 4: Długi Ogon VRU (Motor, Rower, Rider - ultrakrytyczne i zwarte bryły)
        if p_glob_pct < 0.3 and pres_pct < 19.0 and area_pct < 2.3:
            multipliers[t_id] = 2.0
            print(f"[{name:15s}] Długi Ogon VRU   | P_glob: {p_glob_pct:4.2f}% -> Boost: x2.0")
            continue
            
        # Reszta rzadkiej infrastruktury (np. Mury, Płoty)
        multipliers[t_id] = 1.0
        print(f"[{name:15s}] Masywna Infra    | Area: {area_pct:4.1f}%   -> Boost: x1.5")

    # 4. APLIKACJA
    final_weights = base_weights * multipliers

    print("\n=================================================================")
    print(" ---> GOTOWY TENSOR DO FUNKCJI STRATY (Loss Init):")
    print("=================================================================")
    formatted_weights = ", ".join([f"{w:.4f}" for w in final_weights])
    print("self.weight = torch.tensor([")
    print(f"    {formatted_weights}")
    print("])")

if __name__ == "__main__":
    dataset = CS(root="cityscapes", split="train", mode="fine", target_type="semantic",
                 transform=T.ToTensor(), target_transform=lambda mask: torch.from_numpy(np.array(mask, dtype=np.int64)).long())
    # Skanujemy cały zbiór, żeby wyliczyć perfekcyjne statystyki
    loader = DataLoader(dataset, batch_size=8, num_workers=8)
    
    cp, cip, tp, ti = scan_dataset(loader)
    generate_intelligent_weights(cp, cip, tp, ti, 1024 * 2048)