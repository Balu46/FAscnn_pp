import os
import argparse
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision.datasets import Cityscapes
import torchvision.transforms as T
import torch.nn.functional as F
import sys

# Dodajemy bieżący katalog do sys.path aby móc importować z src
sys.path.append(os.getcwd())

# Importy z Twojego projektu
try:
    from src.model_architecture.model_factory import ModelBuilder
    # Usunięto _build_id_to_trainid_lut, ponieważ ewaluacja wymaga mapowania w drugą stronę
    from src.utils.test import _build_model_with_weights, _select_logits
    from src.utils.ablation import AblationConfig
except ImportError:
    print("Błąd: Nie można zaimportować modułów z 'src'. Upewnij się, że uruchamiasz skrypt z głównego folderu projektu.")
    sys.exit(1)

# Normalizacja zgodna z config_example.yaml
MEAN = [0.286, 0.325, 0.283]
STD = [0.176, 0.181, 0.177]

def main():
    parser = argparse.ArgumentParser(description="Zapisuje maski z modelu dla Cityscapes")
    parser.add_argument("--model", type=str, default="FAscnn_pp_V29", help="Nazwa modelu (np. FAscnn_pp_V29, ENet)")
    parser.add_argument("--weights", type=str, required=True, help="Ścieżka do wag (.pt)")
    parser.add_argument("--output", type=str, default="test_masks", help="Folder wyjściowy")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Split danych")
    parser.add_argument("--image-size", nargs=2, type=int, default=[1024, 2048], help="Rozmiar wejściowy (H W)")
    parser.add_argument("--cityscapes-root", type=str, default="./cityscapes", help="Ścieżka do Cityscapes")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output, exist_ok=True)
    
    print(f"Używam urządzenia: {device}")
    
    # Ładowanie modelu
    print(f"Ładowanie modelu {args.model} z {args.weights}...")
    try:
        model, num_classes = _build_model_with_weights(
            args.model, device, ablation_cfg=AblationConfig(), patch_cfg={}, weights_path=args.weights
        )
    except Exception as e:
        print(f"Błąd podczas ładowania modelu: {e}")
        return
        
    model.eval()
    
    # Transformacje wejściowe
    H, W = args.image_size
    img_tf = T.Compose([
        T.Resize((H, W), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])
    
    # Inicjalizacja zbioru danych
    try:
        dataset = Cityscapes(
            root=args.cityscapes_root,
            split=args.split,
            mode='fine',
            target_type='semantic',
            transform=img_tf
        )
    except Exception as e:
        print(f"Błąd podczas ładowania zbioru Cityscapes: {e}")
        return

    # SŁOWNIK MAPOWANIA: 19 klas (trainIds) -> oryginalne labelIds Cityscapes
    trainid_to_labelid = {
        0: 7,   # road
        1: 8,   # sidewalk
        2: 11,  # building
        3: 12,  # wall
        4: 13,  # fence
        5: 17,  # pole
        6: 19,  # traffic light
        7: 20,  # traffic sign
        8: 21,  # vegetation
        9: 22,  # terrain
        10: 23, # sky
        11: 24, # person
        12: 25, # rider
        13: 26, # car
        14: 27, # truck
        15: 28, # bus
        16: 31, # train
        17: 32, # motorcycle
        18: 33  # bicycle
    }
    
    # Tablica LUT w numpy dla natychmiastowego mapowania całej maski
    mapping = np.zeros(256, dtype=np.uint8)
    for train_id, label_id in trainid_to_labelid.items():
        mapping[train_id] = label_id
    
    print(f"Przetwarzanie {len(dataset)} obrazów ze splitu '{args.split}'...")
    
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            # Ze zbioru Cityscapes w trybie 'test' zwracany jest (img, None)
            img, _ = dataset[i]
            img = img.unsqueeze(0).to(device)
            
            # Pobranie oryginalnej nazwy pliku i usunięcie '_leftImg8bit' (wymóg serwera)
            img_path = dataset.images[i]
            basename = os.path.basename(img_path).replace('_leftImg8bit.png', '')
            
            # Inferencja
            outputs = _select_logits(model(img))
            
            # Skalowanie z powrotem do 1024x2048, jeśli model zwraca mniejszą rozdzielczość
            if outputs.shape[-2:] != (1024, 2048):
                 outputs = F.interpolate(outputs, size=(1024, 2048), mode='bilinear', align_corners=False)
            
            # Pobranie najbardziej prawdopodobnych klas (0-18)
            pred = torch.argmax(outputs, dim=1).squeeze(0)
            
            # Przeniesienie na CPU jako uint8
            mask_np = pred.cpu().numpy().astype(np.uint8)
            
            # Mapowanie klas tylko, jeśli model zwrócił 19 klas (trainIds)
            if num_classes == 19:
                mask_np = mapping[mask_np]
            
            # Zapis maski jako 8-bitowy obraz grayscale z oryginalną nazwą bazową
            mask_img = Image.fromarray(mask_np)
            mask_img.save(os.path.join(args.output, f"{basename}.png"))

    print(f"\nGotowe! Maski zapisano w folderze: {os.path.abspath(args.output)}")
    print("Pamiętaj: Skompresuj zawartość folderu (pliki .png) bezpośrednio do pliku .zip, bez tworzenia głównego folderu wewnątrz archiwum.")

if __name__ == "__main__":
    main()