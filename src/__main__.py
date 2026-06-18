"""
Main entry point for FAscnn_pp Fast Segmentation training and testing.
Usage:
  python -m src train --config config.yaml
  python -m src test --config config.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Optional, Tuple

import torch

try:
    import yaml
except ImportError:  # pragma: no cover - handled at runtime
    yaml = None

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# -----------------------------
# Config helpers
# -----------------------------

def _load_yaml_config(path: str) -> dict:
    if not path:
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML is required to read config files. Install with: pip install pyyaml")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping, got: {type(data).__name__}")
    return data


def _get_cfg(cfg: dict, *keys, default=None):
    cur: Any = cfg
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    if cur is None:
        return default
    return cur


def _coalesce(cli_value, cfg_value, default):
    if cli_value is not None:
        return cli_value
    if isinstance(cfg_value, str) and not cfg_value.strip():
        return default
    if cfg_value is None:
        return default
    return cfg_value


def _normalize_optional_str(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return value


def _parse_tuple2_int(value, default: Optional[Tuple[int, int]], name: str) -> Optional[Tuple[int, int]]:
    if value is None:
        return default
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return (int(value[0]), int(value[1]))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid {name} values: {value}") from exc
    raise ValueError(f"{name} must be a list/tuple of 2 numbers, got: {value}")


def _parse_tuple2_float(value, default: Optional[Tuple[float, float]], name: str) -> Optional[Tuple[float, float]]:
    if value is None:
        return default
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return (float(value[0]), float(value[1]))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid {name} values: {value}") from exc
    raise ValueError(f"{name} must be a list/tuple of 2 numbers, got: {value}")


def _parse_tuple3(value, default: Optional[Tuple[float, float, float]], name: str) -> Optional[Tuple[float, float, float]]:
    if value is None:
        return default
    if isinstance(value, (list, tuple)) and len(value) == 3:
        try:
            return (float(value[0]), float(value[1]), float(value[2]))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid {name} values: {value}") from exc
    raise ValueError(f"{name} must be a list/tuple of 3 numbers, got: {value}")


# -----------------------------
# Model + ablation helpers
# -----------------------------

def _ablation_spec_from_cfg(cfg: dict) -> str:
    if not isinstance(cfg, dict):
        return ""
    use_fa1 = cfg.get("use_fa1", True)
    use_fa2 = cfg.get("use_fa2", True)
    use_fa3 = cfg.get("use_fa3", True)
    use_second_branch = cfg.get("use_second_branch", True)

    tokens = []
    if not use_second_branch:
        tokens.append("no_branch")

    if not (use_fa1 and use_fa2 and use_fa3):
        if not use_fa1 and not use_fa2 and not use_fa3:
            tokens.append("no_attn")
        else:
            if not use_fa1:
                tokens.append("no_fa1")
            if not use_fa2:
                tokens.append("no_fa2")
            if not use_fa3:
                tokens.append("no_fa3")

    return ",".join(tokens)


def _available_models():
    from src.model_architecture.model_factory import ModelBuilder

    builder = ModelBuilder(num_classes=1, device="cpu")
    return sorted(builder.base_registry.keys())


def _normalize_model_name(value):
    if not isinstance(value, str):
        return value
    key = value.strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    mapping = {
        "fastscnn": "FastSCNN",
        "enet": "ENet",
        "enetv2": "ENetv2",
        "enetv3": "ENetv3",
        "fascnn_ppv3": "FAscnn_pp_V3",
        "fascnn_ppv6": "FAscnn_pp_V6",
        "fascnn_ppv11": "FAscnn_pp_V11",
        "fascnn_ppv12": "FAscnn_pp_V12",
        "fascnn_ppv13": "FAscnn_pp_V13",
    }
    return mapping.get(key, value)


def _is_all_models(value) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.strip().lower().replace("-", " ").replace("_", " ")
    return normalized in {"all", "all models", "all model"}


def _expand_model_names(model_name: str):
    if _is_all_models(model_name):
        return _available_models()
    return [_normalize_model_name(model_name)]


def _resolve_save_dir(save_dir: Optional[str], model_name: str, crop_size: Optional[Tuple[int, int]] = None) -> str:
    if crop_size:
        h, w = crop_size
        if (h, w) != (1024, 2048):
            model_name = f"{model_name}_{h}x{w}"

    if not save_dir:
        return os.path.join("checkpoints", model_name)
    if "{model}" in save_dir:
        return save_dir.format(model=model_name)
    return save_dir


# -----------------------------
# Argument parser
# -----------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FAscnn_pp Fast Segmentation - Training and Testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Train with default settings:
    python -m src train --config config.yaml

  Train with custom parameters:
    python -m src train --config config.yaml --model FAscnn_pp_V13 --num-epochs 50

  Test the model:
    python -m src test --config config.yaml
        """,
    )

    parser.add_argument("mode", choices=["train", "test"], help="Mode to run: train or test")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file (optional)")

    # Common
    parser.add_argument("--model", type=str, default=None, help="Model architecture (default: from config)")
    parser.add_argument("--num-classes", type=int, default=None, help="Number of output classes")
    parser.add_argument("--device", type=str, default=None, choices=["auto", "cuda", "cpu"], help="Device to use")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=None, help="Number of data loader workers")

    # Training
    parser.add_argument("--cityscapes-root", type=str, default=None, help="Path to Cityscapes dataset")
    parser.add_argument("--num-epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=None, help="SGD momentum")
    parser.add_argument("--weight-decay", type=float, default=None, help="Weight decay")
    parser.add_argument("--amp", action="store_true", default=None, help="Enable AMP training")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="Disable AMP training")
    parser.add_argument("--use-lovasz", action="store_true", default=None, help="Use Lovasz-Softmax loss")
    parser.add_argument("--no-lovasz", dest="use_lovasz", action="store_false", help="Disable Lovasz-Softmax loss")
    parser.add_argument("--is-finetune", action="store_true", default=None, help="Enable fine-tuning mode")
    parser.add_argument("--no-is-finetune", dest="is_finetune", action="store_false", help="Disable fine-tuning mode")
    parser.add_argument("--ignore-index", type=int, default=None, help="Ignore index for loss")
    parser.add_argument("--log-every", type=int, default=None, help="Log every N iterations")
    parser.add_argument("--eval-every-epochs", type=int, default=None, help="Validate every N epochs")
    parser.add_argument("--warmup-epochs", type=int, default=None, help="Number of warmup epochs")
    parser.add_argument("--save-dir", type=str, default=None, help="Checkpoint output directory")
    parser.add_argument("--resume-path", type=str, default=None, help="Resume checkpoint path")
    parser.add_argument("--poly-power", type=float, default=None, help="PolyLR power")
    parser.add_argument("--aug", action="store_true", default=None, help="Enable training augmentation")
    parser.add_argument("--no-aug", dest="aug", action="store_false", help="Disable training augmentation")
    parser.add_argument("--crop-size", nargs=2, type=int, default=None, metavar=("H", "W"), help="Train crop size")
    parser.add_argument("--size", nargs=2, type=int, default=None, metavar=("H", "W"), help="Train image size")
    parser.add_argument("--scale-range", nargs=2, type=float, default=None, metavar=("MIN", "MAX"), help="Scale range")
    parser.add_argument("--flip-p", type=float, default=None, help="Flip probability")
    parser.add_argument("--noise-std", type=float, default=None, help="Noise std")
    parser.add_argument("--noise-p", type=float, default=None, help="Noise probability")
    parser.add_argument("--brightness", type=float, default=None, help="ColorJitter brightness")
    parser.add_argument("--mean", nargs=3, type=float, default=None, metavar=("R", "G", "B"), help="Normalize mean")
    parser.add_argument("--std", nargs=3, type=float, default=None, metavar=("R", "G", "B"), help="Normalize std")

    # EMA
    parser.add_argument("--ema", dest="use_ema", action="store_true", default=None, help="Enable EMA")
    parser.add_argument("--no-ema", dest="use_ema", action="store_false", help="Disable EMA")
    parser.add_argument("--ema-decay", type=float, default=None, help="EMA decay factor")

    # Knowledge Distillation (KD)
    parser.add_argument("--kd", dest="use_kd", action="store_true", default=None, help="Enable Knowledge Distillation")
    parser.add_argument("--no-kd", dest="use_kd", action="store_false", help="Disable Knowledge Distillation")

    # ClassMix
    parser.add_argument("--classmix", dest="use_classmix", action="store_true", default=None, help="Enable ClassMix")
    parser.add_argument("--no-classmix", dest="use_classmix", action="store_false", help="Disable ClassMix")
    parser.add_argument("--classmix-prob", type=float, default=None, help="ClassMix probability")

    # Class Weights
    parser.add_argument("--class-weights-type", type=str, choices=["none", "standard", "boosted"], default=None, help="Type of class weights")

    # Copy-Paste
    parser.add_argument("--copy-paste-prob", type=float, default=None, help="Copy-Paste probability")
    parser.add_argument("--endgame", dest="use_endgame", action="store_true", default=None, help="Enable endgame phase")
    parser.add_argument("--no-endgame", dest="use_endgame", action="store_false", help="Disable endgame phase")
    parser.add_argument("--endgame-threshold", type=float, default=None, help="Progress threshold to start endgame (0.0 to 1.0, e.g. 0.85)")

    # Test
    parser.add_argument("--image-size", nargs=2, type=int, default=None, metavar=("H", "W"), help="Test image size")
    parser.add_argument("--save-vis-per-batch", type=int, default=None, help="Visualizations per batch")
    parser.add_argument("--warmup-iters", type=int, default=None, help="Warmup iterations for timing")
    parser.add_argument("--ablation", type=str, default=None, help="Ablation spec for FAscnn_pp_V13")
    parser.add_argument("--weights-path", type=str, default=None, help="Path to model weights (override default resolution)")
    parser.add_argument("--rescale-mask", action="store_true", default=None, help="Rescale mask to match image size if not full resolution")
    parser.add_argument("--no-rescale-mask", dest="rescale_mask", action="store_false", help="Do not rescale mask")
    parser.add_argument("--run-name", type=str, default=None, help="Custom folder name for saving test results")

    # Patch
    parser.add_argument("--patch", dest="use_patching", action="store_true", default=None, help="Enable patching")
    parser.add_argument("--no-patch", dest="use_patching", action="store_false", help="Disable patching")
    parser.add_argument("--tile-size", nargs=2, type=int, default=None, metavar=("H", "W"), help="Patch tile size")
    parser.add_argument("--overlap", nargs=2, type=int, default=None, metavar=("H", "W"), help="Patch overlap")

    # Pretrained (FastSCNN)
    parser.add_argument("--pretrained", dest="pretrained", action="store_true", default=None, help="Enable pretrained FastSCNN")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false", help="Disable pretrained FastSCNN")
    parser.add_argument("--pretrained-dataset", type=str, default=None, help="Pretrained dataset alias")
    parser.add_argument("--pretrained-root", type=str, default=None, help="Pretrained weights root")
    parser.add_argument("--pretrained-map-cpu", dest="pretrained_map_cpu", action="store_true", default=None, help="Map pretrained weights to CPU")

    parser.set_defaults(
        amp=None,
        use_lovasz=None,
        is_finetune=None,
        aug=None,
        use_patching=None,
        pretrained=None,
        pretrained_map_cpu=None,
        rescale_mask=None,
        use_ema=None,
        use_kd=None,
        use_classmix=None,
        class_weights_type=None,
        use_endgame=None,
    )

    return parser


# -----------------------------
# Resolution helpers
# -----------------------------

def _resolve_ablation(args, cfg, cfg_test) -> str:
    cfg_ablation = _get_cfg(cfg_test, "ablation", default=None)
    if cfg_ablation is None:
        cfg_ablation = _get_cfg(cfg, "ablation", default=None)
    if isinstance(cfg_ablation, dict):
        cfg_ablation = _ablation_spec_from_cfg(cfg_ablation)
    return _coalesce(args.ablation, cfg_ablation, "")

def _resolve_ablation_train(args, cfg_training) -> str:
    cfg_ablation = _get_cfg(cfg_training, "ablation", default=None)
    if isinstance(cfg_ablation, dict):
        cfg_ablation = _ablation_spec_from_cfg(cfg_ablation)
    return _coalesce(args.ablation, cfg_ablation, "")


def _resolve_patch_cfg(args, cfg) -> dict:
    cfg_patch = _get_cfg(cfg, "patch", default={}) or {}
    patch_use = _coalesce(args.use_patching, cfg_patch.get("use_patching"), False)
    patch_tile = _parse_tuple2_int(
        _coalesce(args.tile_size, cfg_patch.get("tile_size"), None),
        default=(64, 64),
        name="tile_size",
    )
    patch_overlap = _parse_tuple2_int(
        _coalesce(args.overlap, cfg_patch.get("overlap"), None),
        default=(8, 8),
        name="overlap",
    )
    return {
        "use_patching": bool(patch_use),
        "tile_size": patch_tile,
        "overlap": patch_overlap,
    }


def _resolve_pretrained_cfg(args, cfg, cfg_model) -> dict:
    cfg_pre = _get_cfg(cfg, "pretrained", default=None)
    if cfg_pre is None:
        cfg_pre = _get_cfg(cfg_model, "pretrained", default={})
    if not isinstance(cfg_pre, dict):
        cfg_pre = {}

    return {
        "enabled": bool(_coalesce(args.pretrained, cfg_pre.get("enabled"), False)),
        "dataset": _coalesce(args.pretrained_dataset, cfg_pre.get("dataset"), "citys"),
        "root": _coalesce(args.pretrained_root, cfg_pre.get("root"), "./weights"),
        "map_cpu": bool(_coalesce(args.pretrained_map_cpu, cfg_pre.get("map_cpu"), False)),
    }


# -----------------------------
# Train / test runners
# -----------------------------

def _run_train(args, cfg_training, model_names, num_classes, patch_cfg, pretrained_cfg, ablation_str) -> None:
    import torch
    from src.data.Cityscape import CityscapesDataModule
    from src.model_architecture.model_factory import ModelBuilder, PretrainedCfg
    from src.utils.train import train_main
    from src.utils.train_config import TrainCfg

    if args.device is not None or _get_cfg(cfg_training, "device", default=None) is not None:
        print(
            "Note: training device is chosen automatically inside src/utils/train.py; --device is ignored.",
            file=sys.stderr,
        )
    device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cityscapes_root = _coalesce(args.cityscapes_root, _get_cfg(cfg_training, "cityscapes_root", default=None), None)
    if cityscapes_root is None:
        cityscapes_root = _get_cfg(cfg_training, "dataset_root", default="./cityscapes")

    crop_size = _parse_tuple2_int(
        _coalesce(args.crop_size, _get_cfg(cfg_training, "crop_size", default=None), None),
        default=(512, 1024),
        name="crop_size",
    )
    scale_range = _parse_tuple2_float(
        _coalesce(args.scale_range, _get_cfg(cfg_training, "scale_range", default=None), None),
        default=(0.5, 2.0),
        name="scale_range",
    )
    flip_p = float(_coalesce(args.flip_p, _get_cfg(cfg_training, "flip_p", default=None), 0.5))
    noise_std = float(_coalesce(args.noise_std, _get_cfg(cfg_training, "noise_std", default=None), 0.05))
    noise_p = float(_coalesce(args.noise_p, _get_cfg(cfg_training, "noise_p", default=None), 0.5))
    brightness = float(_coalesce(args.brightness, _get_cfg(cfg_training, "brightness", default=None), 0.4))
    mean = _parse_tuple3(
        _coalesce(args.mean, _get_cfg(cfg_training, "mean", default=None), None),
        default=(0.485, 0.456, 0.406),
        name="mean",
    )
    std = _parse_tuple3(
        _coalesce(args.std, _get_cfg(cfg_training, "std", default=None), None),
        default=(0.229, 0.224, 0.225),
        name="std",
    )

    size_arg = _parse_tuple2_int(
        _coalesce(args.size, _get_cfg(cfg_training, "size", default=None), None),
        default=(1024, 2048),
        name="size",
    )

    for name in model_names:
        save_dir = _resolve_save_dir(
            _coalesce(args.save_dir, _get_cfg(cfg_training, "save_dir", default=None), None),
            name,
            crop_size=size_arg  # Use size for directory naming
        )
        resume_path = _normalize_optional_str(
            _coalesce(args.resume_path, _get_cfg(cfg_training, "resume_path", default=None), None)
        )

        train_cfg = TrainCfg(
            num_epochs=int(_coalesce(args.num_epochs, _get_cfg(cfg_training, "num_epochs", default=None), 50)),
            lr=float(_coalesce(args.lr, _get_cfg(cfg_training, "lr", default=None), 0.045)),
            momentum=float(_coalesce(args.momentum, _get_cfg(cfg_training, "momentum", default=None), 0.9)),
            weight_decay=float(_coalesce(args.weight_decay, _get_cfg(cfg_training, "weight_decay", default=None), 4e-5)),
            batch_size=int(_coalesce(args.batch_size, _get_cfg(cfg_training, "batch_size", default=None), 12)),
            num_workers=int(_coalesce(args.num_workers, _get_cfg(cfg_training, "num_workers", default=None), 4)),
            amp=bool(_coalesce(args.amp, _get_cfg(cfg_training, "amp", default=None), False)),
            use_lovasz=bool(_coalesce(args.use_lovasz, _get_cfg(cfg_training, "use_lovasz", default=None), False)),
            is_finetune=bool(_coalesce(args.is_finetune, _get_cfg(cfg_training, "is_finetune", default=None), False)),
            ignore_index=int(_coalesce(args.ignore_index, _get_cfg(cfg_training, "ignore_index", default=None), 255)),
            log_every=int(_coalesce(args.log_every, _get_cfg(cfg_training, "log_every", default=None), 50)),
            eval_every_epochs=int(
                _coalesce(args.eval_every_epochs, _get_cfg(cfg_training, "eval_every_epochs", default=None), 1)
            ),
            warmup_epochs=int(
                _coalesce(args.warmup_epochs, _get_cfg(cfg_training, "warmup_epochs", default=None), 3)
            ),
            save_dir=save_dir,
            resume_path=resume_path,
            poly_power=float(_coalesce(args.poly_power, _get_cfg(cfg_training, "poly_power", default=None), 0.9)),
            aug=bool(_coalesce(args.aug, _get_cfg(cfg_training, "aug", default=None), True)),
            size=size_arg,
            use_ema=bool(_coalesce(args.use_ema, _get_cfg(cfg_training, "use_ema", default=True), True)),
            ema_decay=float(_coalesce(args.ema_decay, _get_cfg(cfg_training, "ema_decay", default=0.999), 0.999)),
            use_kd=bool(_coalesce(args.use_kd, _get_cfg(cfg_training, "use_kd", default=False), False)),
            use_classmix=bool(_coalesce(args.use_classmix, _get_cfg(cfg_training, "use_classmix", default=False), False)),
            classmix_prob=float(_coalesce(args.classmix_prob, _get_cfg(cfg_training, "classmix_prob", default=0.5), 0.5)),
            class_weights_type=str(_coalesce(args.class_weights_type, _get_cfg(cfg_training, "class_weights_type", default="none"), "none")),
            copy_paste_prob=float(_coalesce(args.copy_paste_prob, _get_cfg(cfg_training, "copy_paste_prob", default=0.0), 0.0)),
            use_endgame=bool(_coalesce(args.use_endgame, _get_cfg(cfg_training, "use_endgame", default=True), True)),
            endgame_threshold=float(_coalesce(args.endgame_threshold, _get_cfg(cfg_training, "endgame_threshold", default=0.85), 0.85)),
        )

        pretrained_cfg_obj = PretrainedCfg(
            enabled=pretrained_cfg["enabled"],
            dataset=pretrained_cfg["dataset"],
            root=pretrained_cfg["root"],
            map_cpu=pretrained_cfg["map_cpu"],
        )

        from src.utils.ablation import parse_ablation_spec
        ablation_cfg = parse_ablation_spec(ablation_str)

        builder = ModelBuilder(
            num_classes=num_classes,
            device=device_obj,
            patch_cfg=patch_cfg,
            pretrained_cfg=pretrained_cfg_obj,
            ablation_cfg=ablation_cfg,
        )
        model = builder.build(name)

        dm = CityscapesDataModule(
            city_root=cityscapes_root,
            cfg=train_cfg,
            crop_size=crop_size,
            scale_range=scale_range,
            flip_p=flip_p,
            noise_std=noise_std,
            noise_p=noise_p,
            brightness=brightness,
            mean=mean,
            std=std,
        )
        train_loader, val_loader = dm.make_loaders()

        print(f"Starting training: model={name} | epochs={train_cfg.num_epochs} | batch={train_cfg.batch_size}")
        train_main(train_cfg, model, train_loader, val_loader, num_classes=num_classes)


def _run_test(args, cfg_training, cfg_test, model_names, ablation, patch_cfg, run_name) -> None:
    from src.utils.test import test_main
    
    torch.set_num_threads(int(1))
    torch.set_num_interop_threads(1)
    

    test_device = _coalesce(args.device, _get_cfg(cfg_test, "device", default=None), "auto")
    test_batch_size = int(_coalesce(args.batch_size, _get_cfg(cfg_test, "batch_size", default=None), 4))
    test_num_workers = int(_coalesce(args.num_workers, _get_cfg(cfg_test, "num_workers", default=None), 4))
    test_image_size = _parse_tuple2_int(
        _coalesce(args.image_size, _get_cfg(cfg_test, "image_size", default=None), _get_cfg(cfg_test, "size", default=None)),
        default=(1024, 2048),
        name="image_size",
    )
    test_cityscapes_root = _coalesce(args.cityscapes_root, _get_cfg(cfg_test, "cityscapes_root", default=None), None)
    if test_cityscapes_root is None:
        test_cityscapes_root = _get_cfg(cfg_training, "cityscapes_root", default=None)
    if test_cityscapes_root is None:
        test_cityscapes_root = _get_cfg(cfg_training, "dataset_root", default=None)

    save_vis_per_batch = int(_coalesce(args.save_vis_per_batch, _get_cfg(cfg_test, "save_vis_per_batch", default=None), 2))
    warmup_iters = int(_coalesce(args.warmup_iters, _get_cfg(cfg_test, "warmup_iters", default=None), 10))
    weights_path = _coalesce(args.weights_path, _get_cfg(cfg_test, "weights_path", default=None), None)
    rescale_mask = bool(_coalesce(args.rescale_mask, _get_cfg(cfg_test, "rescale_mask", default=None), True))
    use_ema = bool(_coalesce(args.use_ema, _get_cfg(cfg_test, "use_ema", default=True), True))

    for name in model_names:
        ablation_for_model = ablation if name in ["FAscnn_pp_V13", "FAscnn_pp_V18"] else None
        if ablation and name not in ["FAscnn_pp_V13", "FAscnn_pp_V18"]:
            print(
                f"Note: ablation is only supported for FAscnn_pp_V13 and V18; ignoring for {name}.",
                file=sys.stderr,
            )
        test_main(
            batch_size=test_batch_size,
            model_name=name,
            device=test_device,
            ablation=ablation_for_model,
            patch_cfg=patch_cfg,
            image_size=test_image_size,
            cityscapes_root=test_cityscapes_root,
            num_workers=test_num_workers,
            save_vis_per_batch=save_vis_per_batch,
            warmup_iters=warmup_iters,
            weights_path=weights_path,
            rescale_mask=rescale_mask,
            use_ema=use_ema,
            run_name=run_name,
        )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    cfg = _load_yaml_config(args.config) if args.config else {}
    cfg_training = _get_cfg(cfg, "training", default={}) or {}
    cfg_test = _get_cfg(cfg, "test", default={}) or {}
    cfg_model = _get_cfg(cfg, "model", default={}) or {}

    model_name = _coalesce(args.model, _get_cfg(cfg_model, "name", default=None), "FAscnn_pp_V13")
    model_names = _expand_model_names(model_name)
    num_classes = _coalesce(args.num_classes, _get_cfg(cfg_model, "num_classes", default=None), 19)

    ablation = _resolve_ablation(args, cfg, cfg_test)
    patch_cfg = _resolve_patch_cfg(args, cfg)
    pretrained_cfg = _resolve_pretrained_cfg(args, cfg, cfg_model)

    if args.mode == "train":
        ablation_train = _resolve_ablation_train(args, cfg_training)
        _run_train(args, cfg_training, model_names, num_classes, patch_cfg, pretrained_cfg, ablation_train)
    else:
        _run_test(args, cfg_training, cfg_test, model_names, ablation, patch_cfg, args.run_name)


if __name__ == "__main__":
    main()
