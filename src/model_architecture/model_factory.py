from model_architecture.BiSeNet import BiSeNetV2
from src.model_architecture.FAscnn_pp.FAscnn_pp_V13 import FAscnn_pp_V13, FAscnn_pp_V14, FAscnn_pp_V15, FAscnn_pp_V16
from src.model_architecture.FAscnn_pp.FAscnn_pp_V17 import FAscnn_pp_V17, FAscnn_pp_V18
from src.data.patches import model_patched
from src.model_architecture.ENET_plus import *
from src.model_architecture.ENET import ENet
from src.model_architecture.FAST_SCNN import FastSCNN
from src.model_architecture.FAscnn_pp.FAscnn_pp import FAscnn_pp_V6, FAscnn_pp_V3
from src.model_architecture.FAscnn_pp.FAscnn_pp_V11 import FAscnn_pp_V11, FAscnn_pp_V12
import os
import torch
from dataclasses import dataclass


_FAST_SCNN_ACRONYMS = {
    'pascal_voc': 'voc',
    'pascal_aug': 'voc',
    'ade20k': 'ade',
    'coco': 'coco',
    'citys': 'citys',
}



@dataclass
class PretrainedCfg:
    enabled: bool = False
    dataset: str = "citys"
    root: str = "./weights"
    map_cpu: bool = False

class ModelBuilder:
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 19,
        device = None,
        ablation_cfg=None,
        patch_cfg=None,
        pretrained_cfg=None,
    ):
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.device = device
        self.ablation_cfg = ablation_cfg
        self.patch_cfg = patch_cfg or {}
        self.pretrained_cfg = pretrained_cfg

        self.base_registry = {
            # --- klasyczne ---
            "ENet": self._build_enet,
            "ENetv2": self._build_enetv2,
            "ENetv3": self._build_enetv3,
            "FastSCNN": self._build_fast_scnn,

            # --- FAscnn_pp ---
            "FAscnn_pp_V3": lambda: self._build_fascnn_pp(FAscnn_pp_V3),
            "FAscnn_pp_V6": lambda: self._build_fascnn_pp(FAscnn_pp_V6),
            "FAscnn_pp_V11": lambda: self._build_fascnn_pp(FAscnn_pp_V11),
            "FAscnn_pp_V12": lambda: self._build_fascnn_pp(FAscnn_pp_V12),
            "FAscnn_pp_V13": lambda: self._build_fascnn_pp(FAscnn_pp_V13),
            "FAscnn_pp_V14": lambda: self._build_fascnn_pp(FAscnn_pp_V14),  
            "FAscnn_pp_V15": lambda: self._build_fascnn_pp(FAscnn_pp_V15),  
            "FAscnn_pp_V16": lambda: self._build_fascnn_pp(FAscnn_pp_V16), 
            "FAscnn_pp_V17": lambda: self._build_fascnn_pp(FAscnn_pp_V17),
            "FAscnn_pp_V18": lambda: self._build_fascnn_pp(FAscnn_pp_V18),
        }

    # ---------- BUILDERS ----------

    def _build_fascnn_pp(self, model_cls):
        if model_cls.__name__ == "FAscnn_pp_V18":
            return model_cls(
                in_channels=self.in_channels,
                num_classes=self.num_classes,
                ablation_cfg=self.ablation_cfg,
            )
        return model_cls(
            in_channels=self.in_channels,
            num_classes=self.num_classes,
        )

    def _build_enet(self):
        return ENet(n_class=self.num_classes)

    def _build_enetv2(self):
        return ENetv2(num_classes=self.num_classes, in_channels=self.in_channels )

    def _build_enetv3(self):
        return ENetv3(num_classes=self.num_classes, in_channels=self.in_channels)

    def _build_fast_scnn(self):
        model = FastSCNN(self.num_classes)

        if self.pretrained_cfg and self.pretrained_cfg.enabled:
            self._load_fast_scnn_weights(model)

        return model

    # ---------- PRETRAINED ----------

    def _load_fast_scnn_weights(self, model):
        cfg = self.pretrained_cfg
        acronym = _FAST_SCNN_ACRONYMS[cfg.dataset]

        weight_path = os.path.join(
            cfg.root, f"fast_scnn_{acronym}.pth"
        )

        state = torch.load(
            weight_path,
            map_location="cpu" if cfg.map_cpu else None,
        )
        model.load_state_dict(state)

    # ---------- WRAPPERS ----------

    def _apply_patching(self, model):
        if not self.patch_cfg.get("use_patching", False):
            return model

        return model_patched(
            base_model=model,
            tile_size=self.patch_cfg.get("tile_size", (64, 64)),
            overlap=self.patch_cfg.get("overlap", (8, 8)),
        )

    def apply_wrappers(self, model):
        model = self._apply_patching(model)
        return model

    # ---------- ENTRY ----------

    def build(self, model_name):
        if model_name not in self.base_registry:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Available: {list(self.base_registry.keys())}"
            )

        model = self.base_registry[model_name]()
        model = self.apply_wrappers(model)
        return model.to(self.device)
