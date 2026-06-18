import time
import logging
import torch
import random


class TrainingTimeEstimator:
    def __init__(self, total_epochs, logger=None):
        self.total_epochs = total_epochs
        self.start_time = None
        self.epoch_times = []
        self.logger = logger if logger else logging.getLogger(__name__)

    def start(self):
        self.start_time = time.time()

    def end_epoch(self, epoch_idx):
        epoch_time = time.time() - self.start_time
        self.epoch_times.append(epoch_time)
        
        avg_time = sum(self.epoch_times) / len(self.epoch_times)
        remaining_epochs = self.total_epochs - (epoch_idx + 1)
        eta_seconds = avg_time * remaining_epochs

        print(f"Czas tej epoki: {epoch_time / 60:.2f} min")
        print(f"Średni czas epoki: {avg_time / 60:.2f} min")
        print(f"Szacowany czas do końca: {eta_seconds / 3600:.2f} h")

        # Reset startu
        self.start_time = time.time()

    def total_summary(self):
        total_time = sum(self.epoch_times)
        self.logger.info(f"Trening zakończony. "
                         f"Całkowity czas: {total_time / 3600:.2f} h")


class BatchClassMix:
    """
    Wykonuje operację ClassMix na poziomie Batcha (na tensorach GPU).
    Wybiera losowo rzadkie klasy z obrazu A i wkleja je (wraz z maskami) na obraz B.
    """
    def __init__(self, mix_prob=0.5, ignore_index=255):
        self.mix_prob = mix_prob
        self.ignore_index = ignore_index
        
        # Identyfikatory rzadkich/trudnych klas w Cityscapes (trainId)
        # 3: wall, 4: fence, 5: pole, 6: traffic light, 7: traffic sign
        # 12: rider, 14: truck, 15: bus, 16: train, 17: motorcycle, 18: bicycle
        self.rare_classes = [3, 4, 5, 6, 7, 12, 14, 15, 16, 17, 18]

    def __call__(self, images, targets):
        """
        images: Tensor [B, C, H, W]
        targets: Tensor [B, H, W]
        """
        if random.random() > self.mix_prob:
            return images, targets

        B = images.shape[0]
        if B < 2:
            return images, targets # Batch za mały do miksowania

        # Tworzymy tensor wymieszany, początkowo jako klon oryginalnego
        mixed_images = images.clone()
        mixed_targets = targets.clone()

        # Tworzymy losową permutację indeksów batcha (np. dla B=4: [2, 0, 3, 1])
        # Każdy obraz otrzyma wklejki z obrazu wskazanego przez permutację
        rand_indices = torch.randperm(B, device=images.device)

        for i in range(B):
            source_idx = rand_indices[i]
            target_idx = i

            # Jeśli wylosował samego siebie, pomijamy
            if source_idx == target_idx:
                continue

            source_mask = targets[source_idx]
            
            # Szukamy, czy w obrazie źródłowym są jakieś rzadkie klasy
            cut_mask = torch.zeros_like(source_mask, dtype=torch.bool)
            
            # Z prawdopodobieństwem 50% wybieramy połowę rzadkich klas obecnych na zdjęciu
            present_classes = torch.unique(source_mask)
            present_rare_classes = [c.item() for c in present_classes if c.item() in self.rare_classes]
            
            if len(present_rare_classes) == 0:
                continue # Brak obiektów do wycięcia
                
            # Wybieramy losowe obiekty do wklejenia
            num_to_mix = max(1, len(present_rare_classes) // 2)
            classes_to_mix = random.sample(present_rare_classes, num_to_mix)

            # Tworzymy binarną maskę obszarów, które będziemy wycinać
            for c in classes_to_mix:
                cut_mask = cut_mask | (source_mask == c)

            # --- Aplikacja wycięcia (Copy-Paste) ---
            # Jeśli obrazy są mniejsze niż maski (np. image resize, target original)
            if cut_mask.shape[-2:] != images.shape[-2:]:
                cut_mask_resized = torch.nn.functional.interpolate(
                    cut_mask.float().unsqueeze(0).unsqueeze(0), 
                    size=images.shape[-2:], 
                    mode='nearest'
                ).bool().squeeze(0).squeeze(0)
                cut_mask_img = cut_mask_resized.unsqueeze(0).expand_as(images[source_idx])
            else:
                cut_mask_img = cut_mask.unsqueeze(0).expand_as(images[source_idx])

            # Wklejamy piksele ze źródła na cel tam, gdzie cut_mask jest True
            mixed_images[target_idx] = torch.where(cut_mask_img, images[source_idx], mixed_images[target_idx])
            mixed_targets[target_idx] = torch.where(cut_mask, targets[source_idx], mixed_targets[target_idx])

        return mixed_images, mixed_targets

