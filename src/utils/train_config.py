from dataclasses import dataclass
from typing import Optional, Tuple
# -----------------------------
# Konfiguracja
# -----------------------------
@dataclass
class TrainCfg:
    num_epochs: int = 300
    lr: float = 0.045
    momentum: float = 0.9
    weight_decay: float = 4e-5  

    batch_size: int = 12
    num_workers: int = 4

    amp: bool = False 
    use_lovasz: bool = False
    is_finetune: bool = False
    ignore_index: int = 255

    log_every: int = 50
    eval_every_epochs: int = 1
    warmup_epochs: int = 3

    save_dir: str = "checkpoints"
    resume_path: Optional[str] = None

    # PolyLR 
    poly_power: float = 0.9
    
    aug : bool = True
    size : Tuple[int, int] = (1024, 2048)

    # EMA
    use_ema: bool = True
    ema_decay: float = 0.999

    # Knowledge Distillation
    use_kd: bool = False

    # ClassMix
    use_classmix: bool = False
    classmix_prob: float = 0.5

    # Class Weights
    class_weights_type: str = "none"

    # Copy-Paste
    copy_paste_prob: float = 0.0

    # Endgame (phase to disable CP and Teacher)
    use_endgame: bool = True
    endgame_threshold: float = 0.85
