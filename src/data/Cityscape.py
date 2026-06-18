import multiprocessing

from PIL import Image, ImageOps, ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import Cityscapes as TVCityscapes
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import torch
import numpy as np
import random
from torchvision.datasets import Cityscapes
from torchvision.datasets.cityscapes import Cityscapes as CS
from typing import Optional, Tuple
import os
from torch.utils.data import ConcatDataset

class JointRandomScale:
    def __init__(self, min_scale=0.5, max_scale=2.0):
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, img, mask):
        scale = random.uniform(self.min_scale, self.max_scale)
        w, h = img.size
        new_w, new_h = int(w * scale), int(h * scale)

        img = img.resize((new_w, new_h), Image.BILINEAR)
        mask = mask.resize((new_w, new_h), Image.NEAREST)
        return img, mask

class JointRandomCrop:
    def __init__(self, crop_size=(768, 768), ignore_index=255):
        self.crop_h, self.crop_w = crop_size
        self.ignore_index = ignore_index

    def __call__(self, img, mask):
        w, h = img.size

        pad_w = max(0, self.crop_w - w)
        pad_h = max(0, self.crop_h - h)

        if pad_w > 0 or pad_h > 0:
            img = ImageOps.expand(img, border=(0, 0, pad_w, pad_h), fill=(0, 0, 0))
            mask = ImageOps.expand(mask, border=(0, 0, pad_w, pad_h), fill=self.ignore_index)
            w, h = img.size

        x1 = random.randint(0, w - self.crop_w)
        y1 = random.randint(0, h - self.crop_h)

        img = img.crop((x1, y1, x1 + self.crop_w, y1 + self.crop_h))
        mask = mask.crop((x1, y1, x1 + self.crop_w, y1 + self.crop_h))
        return img, mask

class JointRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            img = TF.hflip(img)
            mask = TF.hflip(mask)
        return img, mask

class RandomChannelNoise:
    """
    Szum per-kanał (RGB) w przestrzeni [0,1] przed Normalize.
    """
    def __init__(self, noise_std=0.05, p=0.5):
        self.noise_std = noise_std
        self.p = p

    def __call__(self, x: torch.Tensor):
        if torch.rand(1).item() < self.p:
            x = x + torch.randn_like(x) * self.noise_std
            x = x.clamp_(0.0, 1.0)
        return x

def joint_train_transform(img, mask):
    img, mask = JointRandomScale(0.5, 2.0)(img, mask)
    img, mask = JointRandomCrop((768, 768), ignore_index=255)(img, mask)
    img, mask = JointRandomHorizontalFlip(0.5)(img, mask)
    return img, mask

import torch
import numpy as np
import random
import multiprocessing
from PIL import Image
from torch.utils.data import Dataset

class CityscapesWithJointAug(Dataset):
    """
    base_ds zwraca (PIL_img, PIL_mask)
    joint_tf działa na (PIL_img, PIL_mask) i robi geometrię
    img_tf działa na PIL_img (fotometryka + ToTensor + Normalize)
    label_tf mapuje PIL_mask -> LongTensor [H,W] z trainId i ignore_index
    """
    def __init__(self, base_ds, joint_tf=None, img_tf=None, label_tf=None, copy_paste_prob=0.0, is_train=False):
        self.base = base_ds
        self.joint_tf = joint_tf
        self.img_tf = img_tf
        self.label_tf = label_tf
        self.is_train = is_train
        
        # --- Ustawienia Copy-Paste ---
        if self.is_train:
            self.copy_paste_prob = multiprocessing.Value('f', copy_paste_prob)
        
        # UWAGA: Używamy oryginalnych ID z Cityscapes (przed zmapowaniem na trainId)
        # 19: traffic light, 20: traffic sign, 25: rider, 32: motorcycle, 33: bicycle
        self.rare_class_ids = np.array([19, 20, 25, 32, 33]) 
        
        # Tarcza ochronna usunięta na rzecz balansowania wag (Loss Function).

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, mask = self.base[idx]  # PIL img, PIL mask

        # 1) Geometria na głównym obrazie (BIORCA)
        if self.joint_tf is not None:
            img, mask = self.joint_tf(img, mask)

        if self.is_train:
            current_prob = self.copy_paste_prob.value
            
            # ================================================================
            # --- CZYSTE COPY-PASTE (DŁUGI OGON)
            # ================================================================
            if current_prob > 0.0 and random.random() < current_prob:
                # Losujemy dawcę
                idx2 = random.randint(0, len(self.base) - 1)
                img2, mask2 = self.base[idx2]
                
                # Geometria na dawcy (żeby wymiary po Cropie się zgadzały)
                if self.joint_tf is not None:
                    img2, mask2 = self.joint_tf(img2, mask2)
                    
                # Przejście na NumPy dla szybkiego maskowania binarnego
                img_np, mask_np = np.array(img), np.array(mask)
                img2_np, mask2_np = np.array(img2), np.array(mask2)
                
                # Szukamy rzadkich klas na obrazie dawcy
                paste_mask = np.isin(mask2_np, self.rare_class_ids)
                
                # Jeśli dawca ma cokolwiek ciekawego, wklejamy to na biorcę!
                if paste_mask.any():
                    img_np[paste_mask] = img2_np[paste_mask]
                    mask_np[paste_mask] = mask2_np[paste_mask]
                    
                # Wracamy do PIL, by reszta potoku (img_tf) zadziałała bez zmian
                img = Image.fromarray(img_np)
                mask = Image.fromarray(mask_np)
            # ================================================================

        # 2) Fotometryka tylko na obrazie (szum, kolory)
        if self.img_tf is not None:
            img = self.img_tf(img)

        # 3) Mapowanie etykiet na końcu (label_id -> train_id)
        if self.label_tf is not None:
            mask = self.label_tf(mask)
        else:
            mask = torch.from_numpy(np.array(mask, dtype=np.int64)).long()

        return img, mask


class JointCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask



class CityscapesDataModule:
    """
    Użycie:
        dm = CityscapesDataModule(CITY_ROOT, cfg, crop_size=(1024,2048))
        train_loader, val_loader = dm.make_loaders()
    """
    def __init__(self,
                 city_root: str,
                 cfg,
                 crop_size: Tuple[int, int] = (512, 1024),
                 scale_range: Tuple[float, float] = (0.5, 2.0),
                 flip_p: float = 0.5,
                 noise_std: float = 0.05,
                 noise_p: float = 0.5,
                 brightness: float = 0.4,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)):
        self.city_root = city_root
        self.cfg = cfg
        self.crop_size = crop_size
        self.scale_range = scale_range
        self.flip_p = flip_p
        self.noise_std = noise_std
        self.noise_p = noise_p
        self.brightness = brightness
        self.mean = mean
        self.std = std

        # sanity checks
        assert os.path.isdir(os.path.join(city_root, "leftImg8bit")), "Brak leftImg8bit"
        assert os.path.isdir(os.path.join(city_root, "gtFine")), "Brak gtFine"

        # labelId -> trainId
        # self.id_to_trainid = {c.id: c.train_id for c in CS.classes}
        self.id_to_trainid = {c.id: c.train_id for c in CS.classes if c.id >= 0}

    def target_transform(self, mask_pil):
        mask = np.array(mask_pil, dtype=np.int64)
        out = np.full_like(mask, self.cfg.ignore_index, dtype=np.int64)
        for label_id, train_id in self.id_to_trainid.items():
            if train_id == -1:
                continue
            out[mask == label_id] = train_id
        return torch.from_numpy(out).long()

    def _build_transforms(self):
        # Image resize transform if size is specified
        img_resize = []
        if hasattr(self.cfg, "size") and self.cfg.size is not None and tuple(self.cfg.size) != (1024, 2048):
            img_resize.append(T.Resize(self.cfg.size, interpolation=T.InterpolationMode.BILINEAR))

        # VAL: zawsze bez augmentacji (tylko normalize i opcjonalny resize zdjęcia)
        img_only_val = T.Compose(img_resize + [
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std),
        ])

        # TRAIN: zależnie od cfg.aug
        if getattr(self.cfg, "aug", False):
            joint_train = JointCompose([
                JointRandomScale(*self.scale_range),
                JointRandomCrop(crop_size=self.crop_size, ignore_index=self.cfg.ignore_index),
                JointRandomHorizontalFlip(p=self.flip_p),
            ])

            img_only_train = T.Compose(img_resize + [
                T.ColorJitter(brightness=self.brightness),  # brightness
                T.ToTensor(),
                RandomChannelNoise(noise_std=self.noise_std, p=self.noise_p),  # channel noise
                T.Normalize(mean=self.mean, std=self.std),
            ])
        else:
            joint_train = None
            img_only_train = T.Compose(img_resize + [
                T.ToTensor(),
                T.Normalize(mean=self.mean, std=self.std),
            ])

        return joint_train, img_only_train, img_only_val

    def make_datasets(self):
        joint_train, img_only_train, img_only_val = self._build_transforms()
        
        if self.cfg.aug:
            print("Using data augmentation for training.")
            train_base = Cityscapes(
                root=self.city_root,
                split="train",
                mode="fine",
                target_type="semantic",
                transform=None,
                target_transform=None,
            )

            val_base = Cityscapes(
                root=self.city_root,
                split="val",
                mode="fine",
                target_type="semantic",
                transform=None,
                target_transform=None,
            )

            train_ds = CityscapesWithJointAug(
                train_base,
                joint_tf=joint_train,
                img_tf=img_only_train,
                label_tf=self.target_transform,
                copy_paste_prob=getattr(self.cfg, 'copy_paste_prob', 0.0),  # <-- Pobierane z configu
                is_train=True
            )

            val_ds = CityscapesWithJointAug(
                val_base,
                joint_tf=None,
                img_tf=img_only_val,
                label_tf=self.target_transform,
                copy_paste_prob=0.0,  # <-- W walidacji absolutnie 0.0!
                is_train=False
            )
        else:
            print("No data augmentation. Using basic Cityscapes datasets with img-only transforms.")
            train_ds = Cityscapes(
                root=self.city_root,
                split="train",
                mode="fine",
                target_type="semantic",
                transform=img_only_train,
                target_transform=self.target_transform,
            )

            val_ds = Cityscapes(
                root=self.city_root,
                split="val",
                mode="fine",
                target_type="semantic",
                transform=img_only_val,
                target_transform=self.target_transform,
            )
        return train_ds, val_ds

    def make_loaders(self):
        train_ds, val_ds = self.make_datasets()

        train_loader = DataLoader(
            train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=2,
            persistent_workers=True  # <-- DODAJ TO
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True  # <-- DODAJ TO
        )

        return train_loader, val_loader