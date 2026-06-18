# --- TARCZA ANTY-DEADLOCKOWA ---
import os
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
# -------------------------------

import logging
import time
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import islice

from src.utils.loss import BiSeNetOhemCELoss
from src.utils.lr_scheduler import PolyLR
from src.utils.utils import BatchClassMix
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from src.utils.train_config import TrainCfg
from src.utils.metrics import fast_confusion_miou, compute_pixel_accuracy
from torch.utils.tensorboard import SummaryWriter
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn, update_bn
from transformers import SegformerForSemanticSegmentation
from torch.nn import functional as F

# =====================================================================
# 1. FUNKCJE POMOCNICZE (Checkpointy)
# =====================================================================
def save_ckpt(path: str,
              model: nn.Module,
              optimizer: optim.Optimizer,
              scheduler: PolyLR,
              scaler: Optional[GradScaler],
              epoch: int,
              global_step: int,
              best_miou: float,
              ema_model: Optional[nn.Module] = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_miou": best_miou,
    }
        
    torch.save(state_dict, path)


def load_ckpt(path: str, model: nn.Module, optimizer: optim.Optimizer, scheduler: PolyLR,
              scaler: Optional[GradScaler], ema_model: Optional[nn.Module] = None,
              is_finetune: bool = False) -> Tuple[int, int, float]:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
        
    # --- [TRYB FINE-TUNINGU] ---
    if is_finetune:
        logging.getLogger(__name__).info("Fine-tuning: Wagi załadowane. Resetuję Optimizer, Scheduler i Epoki!")
        # Zwracamy epoch=0, global_step=0, zachowujemy tylko best_miou dla logów
        return 0, 0, ckpt.get("best_miou", 0.0) 
    
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
        
    return ckpt["epoch"], ckpt["global_step"], ckpt.get("best_miou", 0.0)

# =====================================================================
# 2. PĘTLA JEDNEJ EPOKI
# =====================================================================
def train_one_epoch(model: nn.Module,
                    train_loader: DataLoader,
                    criterion: nn.Module,
                    optimizer: optim.Optimizer,
                    scheduler: PolyLR,
                    device: torch.device,
                    scaler: Optional[GradScaler],
                    cfg: TrainCfg,
                    epoch: int,
                    global_step: int,
                    warmup_iters: int,
                    logger: Optional[logging.Logger] = None,
                    writer: Optional[SummaryWriter] = None,
                    ema_model: Optional[AveragedModel] = None,
                    teacher_model: Optional[nn.Module] = None,
                    is_finetune: bool = False) -> int: 
    model.train()
    

    running_loss = 0.0
    t0 = time.time()
    if logger is None:
        logger = logging.getLogger(__name__)

    # Harmonogram ClassMixa
    disable_classmix_epoch = getattr(cfg, 'disable_classmix_epoch', 200)
    use_classmix_base = getattr(cfg, 'use_classmix', False)
    current_use_classmix = use_classmix_base and (epoch < disable_classmix_epoch)
    
    if use_classmix_base and epoch == disable_classmix_epoch:
        logger.info(f"--- [EPOCH {epoch}] Wyłączam ClassMix (osiągnięto próg disable_classmix_epoch={disable_classmix_epoch}) ---")
        
    for it, (images, targets) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).long()

        optimizer.zero_grad(set_to_none=True)
               
       
                
        # --- FORWARD UCZNIA I OBLICZANIE STRATY ---
        if scaler is not None:
            with autocast('cuda'):
                outputs = model(images) 
                
                # Interpolate outputs to match targets shape if needed
                if isinstance(outputs, dict):
                    for k in outputs.keys():
                        if k in ["main", "aux_context", "aux_spatial", "aux_border", "aux1", "aux2", "aux3", "aux4", "context", "spatial"]:
                            if outputs[k].shape[-2:] != targets.shape[-2:]:
                                outputs[k] = F.interpolate(outputs[k], size=targets.shape[-2:], mode='bilinear', align_corners=False)
                else:
                    if outputs.shape[-2:] != targets.shape[-2:]:
                        outputs = F.interpolate(outputs, size=targets.shape[-2:], mode='bilinear', align_corners=False)

                if model.__type__() in ["FastSCNN", "FAscnn_pp_v18"]:

                    loss_main = criterion(outputs["main"], targets)
                    loss_aux_ctx = criterion(outputs["context"], targets) * 0.4
                    los_aux_spa =  criterion(outputs["spatial"], targets) * 0.4
                    loss = loss_main + loss_aux_ctx + los_aux_spa
                    loss_dict = {}
                
                elif model.__type__() in ["BiSeNetV2"]:
                    loss_main = criterion(outputs["main"], targets)
                    loss_aux1 = criterion(outputs["aux1"], targets) 
                    loss_aux2 = criterion(outputs["aux2"], targets) 
                    loss_aux3 = criterion(outputs["aux3"], targets)  
                    loss_aux4 = criterion(outputs["aux4"], targets)  
                    loss = loss_main + loss_aux1 + loss_aux2 + loss_aux3 + loss_aux4
                    loss_dict = {}
                    
                else:
                    loss, loss_dict = criterion(outputs, targets)
                    loss_dict = {}
                    
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            
            # Interpolate outputs to match targets shape if needed
            if isinstance(outputs, dict):
                for k in outputs.keys():
                    if k in ["main", "aux_context", "aux_spatial", "aux_border", "aux1", "aux2", "aux3", "aux4"]:
                        if outputs[k].shape[-2:] != targets.shape[-2:]:
                            outputs[k] = F.interpolate(outputs[k], size=targets.shape[-2:], mode='bilinear', align_corners=False)
            else:
                if outputs.shape[-2:] != targets.shape[-2:]:
                    outputs = F.interpolate(outputs, size=targets.shape[-2:], mode='bilinear', align_corners=False)

            if model.__type__() in ["FastSCNN", "FAscnn_pp_v18"]:
                loss_main = criterion(outputs["main"], targets)
                loss_aux_ctx = criterion(outputs["cotext"], targets) * 0.4
                los_aux_spa =  criterion(outputs["spatial"], targets) * 0.4
                loss = loss_main + loss_aux_ctx + los_aux_spa
                loss_dict = {}
            elif model.__type__() in ["BiSeNetV2"]:
                loss_main = criterion(outputs["main"], targets)
                loss_aux1 = criterion(outputs["aux1"], targets) 
                loss_aux2 = criterion(outputs["aux2"], targets) 
                loss_aux3 = criterion(outputs["aux3"], targets)  
                loss_aux4 = criterion(outputs["aux4"], targets)  
                loss = loss_main + loss_aux1 + loss_aux2 + loss_aux3 + loss_aux4
                loss_dict = {}
            else:
                loss, loss_dict = criterion(outputs, targets)
                loss_dict = {}
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()


        # --- HARMONOGRAM LEARNING RATE ---
        if global_step < warmup_iters and warmup_iters > 0:
            warmup_start_lr = 0.001
            current_lr = warmup_start_lr + (cfg.lr - warmup_start_lr) * (global_step / warmup_iters)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        else:
            scheduler.step()    
            
        global_step += 1
        
        # --- LOGOWANIE ---
        if writer is not None:
            writer.add_scalar("train/loss_iter", float(loss.item()), global_step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
            for k, v in loss_dict.items():
                if k != "total_loss":
                    writer.add_scalar(f"train_components/{k}", v, global_step)

        running_loss += float(loss.item())
        if (it + 1) % cfg.log_every == 0:
            avg_loss = running_loss / cfg.log_every
            if writer is not None:
                writer.add_scalar("train/loss_avg", avg_loss, global_step)
            running_loss = 0.0
            lr = optimizer.param_groups[0]["lr"]
            dt = time.time() - t0
            
            log_str = f"[epoch {epoch:03d} | iter {it + 1:05d}/{len(train_loader):05d}] loss={avg_loss:.4f} lr={lr:.6f} time={dt:.1f}s"
            if "loss_kd" in loss_dict:
                log_str += f" | kd={loss_dict['loss_kd']:.3f} bnd={loss_dict.get('loss_boundary', 0):.3f}"
            if "loss_feat_ffm" in loss_dict:
                log_str += f" | feat_kd={loss_dict['loss_feat_ffm']:.4f}"
            
            logger.info(log_str)
            t0 = time.time()

    return global_step

# =====================================================================
# 3. GŁÓWNA PĘTLA TRENINGOWA
# =====================================================================
def train_main(cfg: TrainCfg,
               model: nn.Module,
               train_loader: DataLoader,
               val_loader: DataLoader,
               num_classes: int = 19):
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    log_dir = os.path.join(root, "log", "train")
    os.makedirs(log_dir, exist_ok=True)
    model_tag = os.path.basename(os.path.normpath(cfg.save_dir)) or "train"
    
    is_finetune = getattr(cfg, 'is_finetune', False)
    prefix = "finetune" if is_finetune else "train"
    log_path = os.path.join(log_dir, f"{prefix}_{model_tag}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)
    
    tb_dir = os.path.join(root, "log", "tensorboard", f"{prefix}_{model_tag}")
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)


    # --- OPTIMIZER ---
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay
    )
    
    
    if getattr(cfg, 'class_weights_type', 'none') == 'boosted':
        weights = torch.FloatTensor([
            0.8373, 0.9180, 0.8660, 
            1.5517, # (3) Wall: było 1.0345 -> x1.5
            1.5249, # (4) Fence: było 1.0166 -> x1.5
            1.4953, # (5) Pole: było 0.9969 -> x1.5
            0.9754, 1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 
            2.2232, # (12) Rider: było 1.1116 -> x2.0 
            0.9037, 1.0865, 1.0955, 1.0865, 
            2.3058, # (17) Motorcycle: było 1.1529 -> x2.0 
            1.0507
        ])
    elif getattr(cfg, 'class_weights_type', 'none') == 'standard':
        weights = torch.FloatTensor([
            0.8373, 0.9180, 0.8660, 
            1.0345, # (3) Wall
            1.0166, # (4) Fence
            0.9969, # (5) Pole
            0.9754, 1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 
            1.1116, # (12) Rider
            0.9037, 1.0865, 1.0955, 1.0865, 
            1.1529, # (17) Motorcycle
            1.0507
        ])
    else:
        weights = None
    
    use_lovasz = getattr(cfg, 'use_lovasz', False)
    
    # --- CRITERION ---
    
    if model.__type__() in ["BiSeNetV2"]:
        logger.info("Using standard ohem for BiSeNetV2.")
        criterion = BiSeNetOhemCELoss(thresh=0.7, ignore_lb=cfg.ignore_index)
    else: 
        logger.info(f"Using standard CrossEntropyLoss with class weights: {getattr(cfg, 'class_weights_type', 'none')}")
        criterion = nn.CrossEntropyLoss(ignore_index=cfg.ignore_index, weight=weights.to(device) if weights is not None else None)

    # --- SCHEDULER & WARMUP ---
    warmup_epochs = getattr(cfg, 'warmup_epochs', 3)
    if is_finetune:
        warmup_epochs = 0 
        logger.info("Fine-tuning mode active: Warmup disabled.")
        
    warmup_iters = warmup_epochs * len(train_loader)
    max_iters_poly = max(1, (cfg.num_epochs * len(train_loader)) - warmup_iters)
    scheduler = PolyLR(optimizer, max_iters=max_iters_poly, power=cfg.poly_power)
    scaler = GradScaler() if (cfg.amp and device.type == "cuda") else None

    # --- CHECKPOINT RESUME ---
    start_epoch = 0
    global_step = 0
    best_miou = 0.0

    if cfg.resume_path and os.path.isfile(cfg.resume_path):
        start_epoch, global_step, best_miou = load_ckpt(
            cfg.resume_path, model, optimizer, scheduler, scaler, is_finetune=is_finetune
        )
        logger.info("Resumed: epoch=%d, step=%d, best_mIoU=%.4f", start_epoch, global_step, best_miou)

    os.makedirs(cfg.save_dir, exist_ok=True)

    # =====================================================================
    # GŁÓWNA PĘTLA EPOK
    # =====================================================================
    if hasattr(train_loader.dataset, 'copy_paste_prob'):
        train_loader.dataset.copy_paste_prob.value = cfg.copy_paste_prob
    
    for epoch in range(start_epoch, cfg.num_epochs):
        start_time = time.time()   
        current_progress = epoch / max(1, cfg.num_epochs - 1)


        
        # --- CURRICULUM LEARNING: Aktualizacja wag strat ---
        if hasattr(criterion, 'update_weights'):
            lovasz_start = getattr(cfg, 'lovasz_start_epoch', 340)
            criterion.update_weights(epoch, lovasz_start_epoch=lovasz_start, is_finetune=is_finetune)
        
        
        # --- THE ENDGAME: Faza finalna treningu ---
        if cfg.use_endgame and current_progress >= cfg.endgame_threshold and not is_finetune:
            base_prob = cfg.copy_paste_prob
            # 1. Obliczamy, jak daleko jesteśmy w fazie Endgame (wartość od 0.0 do 1.0)
            # Przykładowo: jeśli threshold to 0.85, a current to 0.90, to jesteśmy w 33% Endgame'u.
            endgame_progress = (current_progress - cfg.endgame_threshold) / (1.0 - cfg.endgame_threshold)
            
            # Zabezpieczenie przed wartościami > 1.0 na samym końcu
            endgame_progress = max(0.0, min(1.0, endgame_progress))
            
            # 2. Liniowy spadek prawdopodobieństwa
            new_prob = base_prob * (1.0 - endgame_progress)
            
            # 3. Wrzucenie nowej wartości do pamięci współdzielonej (zabezpieczone przed ujemnymi)
            train_loader.dataset.copy_paste_prob.value = max(0.0, new_prob)
            
            # Logowanie zjawiska (aby widział Pan w terminalu, jak wartość zjeżdża np. 0.28, 0.25, 0.10...)
            logger.info(f"--- [EPOCH {epoch}] THE ENDGAME: Wygaszanie Copy-Paste (Aktualne prob: {new_prob:.4f}) ---")
             
             
        
        # --- TRENING ---
        global_step = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            scaler=scaler,
            cfg=cfg,
            epoch=epoch,
            global_step=global_step,
            warmup_iters=warmup_iters, 
            logger=logger,
            writer=writer,
            is_finetune=is_finetune
        )

        # --- WALIDACJA I ZAPIS ---
        if (epoch + 1) % cfg.eval_every_epochs == 0:
            val_model = model
            
            if val_model.__type__() in ["BiSeNetV2"]:
                # logger.info("Evaluating model (BiSeNetV2) on validation set...")
                val_model.aux_mode = "eval"
            
            miou = fast_confusion_miou(
                model=val_model, val_loader=val_loader,
                num_classes=num_classes, device=device, ignore_index=cfg.ignore_index
            )
            acc = compute_pixel_accuracy(
                model=val_model, val_loader=val_loader,
                device=device, ignore_index=cfg.ignore_index
            )
            
            if val_model.__type__() in ["BiSeNetV2"]:
                val_model.aux_mode = "train"

            if writer is not None:
                writer.add_scalar("val/mIoU", miou, global_step)
                writer.add_scalar("val/acc", acc, global_step)
            
            epoch_time = time.time() - start_time
            logger.info("[epoch %03d] val mIoU=%.4f | val acc=%.4f | epoch_time=%.2f", epoch, miou, acc, epoch_time)

            is_best = miou > best_miou
            if is_best:
                best_miou = miou

            last_name = "finetune_last.pt" if is_finetune else "last.pt"
            best_name = "finetune_best.pt" if is_finetune else "best.pt"

            save_ckpt(os.path.join(cfg.save_dir, last_name), model, optimizer, scheduler, scaler,
                      epoch=epoch + 1, global_step=global_step, best_miou=best_miou)

            if is_best:
                save_ckpt(os.path.join(cfg.save_dir, best_name), model, optimizer, scheduler, scaler,
                          epoch=epoch + 1, global_step=global_step, best_miou=best_miou)

    if writer is not None:
        writer.close()
    logger.info("Done. best mIoU=%.4f", best_miou)