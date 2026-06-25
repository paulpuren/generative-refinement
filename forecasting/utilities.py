import os
import re
import sys
from types import SimpleNamespace
from pathlib import Path

import cv2
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch_ema import ExponentialMovingAverage

from forecasting.data_nskt import NSKTForecast, NSKTForecastEval
from forecasting.diffusion_model import ForecastingDiffusionModel
from src.flex import FLEX
from src.lion import Lion
from src.utilities import materialize_checkpoint_modules, save_metrics


def cal_ssim_like_shanghai(pred, true, data_range=1.0):
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    img1 = pred.astype(float)
    img2 = true.astype(float)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = kernel @ kernel.T
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1 ** 2
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2 ** 2
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1 * mu2
    ssim_map = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean()


def build_model(args):
    if args.model != "FLEXForecast":
        print("forecasting supports model=FLEXForecast only.")
        sys.exit(1)
    encoder, task_encoder, task_encoder_end, decoder = FLEX(
        image_size=args.patch_size,
        in_channels=1,
        out_channels=1,
        model_size=args.flex_model_size,
        mlp_ratio=args.flex_mlp_ratio,
        use_scalar_film=args.use_scalar_film,
        use_spatial_cond=args.use_spatial_cond,
        spatial_cond_scales=args.spatial_cond_scales,
        spatial_cond_mode=args.spatial_cond_mode,
    )
    return ForecastingDiffusionModel(
        encoder=encoder.cuda(),
        decoder=decoder.cuda(),
        task_encoder=task_encoder.cuda(),
        task_encoder_end=task_encoder_end.cuda() if task_encoder_end is not None else None,
        diff_steps=args.time_steps,
        prediction_type=args.prediction_type,
        criterion=torch.nn.L1Loss(),
        dt_normalization_scale=float(max(args.forecast_train_horizon, args.total_interp_steps_train, 1)),
        condition_on_re=args.condition_on_re,
        condition_on_total_interp_steps=args.condition_on_total_interp_steps,
        forecast_baseline=args.forecast_baseline,
        forecast_conditioning_mode=args.forecast_conditioning_mode,
    )


def load_train_objs(args):
    if args.data_name != "forecasting_nskt":
        print("forecasting supports data_name=forecasting_nskt only.")
        sys.exit(1)
    train_set = NSKTForecast(
        patch_size=args.patch_size,
        crop_size=args.crop_size,
        stride=args.stride,
        scratch_dir=args.scratch_dir,
        flag="train",
        reynolds_normalization_scale=args.reynolds_normalization_scale,
        forecast_train_horizon=args.forecast_train_horizon,
    )
    val_set = NSKTForecast(
        patch_size=args.patch_size,
        crop_size=args.crop_size,
        stride=args.stride,
        scratch_dir=args.scratch_dir,
        flag="valid",
        reynolds_normalization_scale=args.reynolds_normalization_scale,
        forecast_train_horizon=args.forecast_train_horizon,
    )
    model = build_model(args)
    ema = ExponentialMovingAverage(model.parameters(), decay=0.999)

    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == "lion":
        optimizer = Lion(model.parameters(), lr=args.learning_rate)
    else:
        print("Only Adam and Lion are supported.")
        sys.exit(1)
    return train_set, val_set, model, optimizer, ema


def load_eval_obj(args):
    eval_set = NSKTForecastEval(
        patch_size=args.patch_size,
        crop_size=args.crop_size,
        stride=args.stride,
        forecast_horizon=args.forecast_horizon,
        re_id=args.re_id,
        scratch_dir=args.scratch_dir,
        reynolds_normalization_scale=args.reynolds_normalization_scale,
    )
    model = build_model(args)
    ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
    return eval_set, model, ema


def load_checkpoint(save_path, model, optimizer, device, load_optimizer_state=True):
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"Unable to load from {save_path}")
    if isinstance(device, int):
        map_location = torch.device(f"cuda:{device}") if torch.cuda.is_available() else torch.device("cpu")
    else:
        map_location = device
    checkpoint = torch.load(save_path, weights_only=True, map_location=map_location)
    materialize_checkpoint_modules(model, checkpoint["model"])
    model.load_state_dict(checkpoint["model"])
    ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
    if "ema" in checkpoint:
        ema.load_state_dict(checkpoint["ema"])
    if load_optimizer_state and optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return model, ema, optimizer, int(checkpoint.get("epoch", 0))


def prepare_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int = 8,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
):
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "pin_memory": True,
        "sampler": DistributedSampler(dataset),
        "shuffle": False,
        "num_workers": num_workers,
        "drop_last": True,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**loader_kwargs)


def get_run_name(args):
    condition_tag = ""
    if not getattr(args, "condition_on_re", True):
        condition_tag += "_noRe"
    reynolds_scale = float(getattr(args, "reynolds_normalization_scale", 1.0))
    if reynolds_scale != 1.0:
        condition_tag += f"_ReNorm{reynolds_scale:g}"
    if not getattr(args, "condition_on_total_interp_steps", True):
        condition_tag += "_noT"
    forecast_baseline = getattr(args, "forecast_baseline", "persistence")
    if forecast_baseline != "persistence":
        condition_tag += f"_baseline{forecast_baseline}"
    prefix = "Model_ft" if getattr(args, "checkpoint_path", "") else "Model"
    return (
        f"{prefix}_{args.model}{condition_tag}_{args.flex_model_size}_mlp{args.flex_mlp_ratio}"
        f"_Data_{args.data_name}_Optim_{args.optimizer}_cosine_lr{args.learning_rate}"
        f"_epoch{args.epochs}_patchsize{args.patch_size}_stride{args.stride}"
        f"_oneStepForecast"
    )


def resolve_eval_checkpoint_path(args):
    checkpoint_path = getattr(args, "checkpoint_path", "")
    original_checkpoint_path = checkpoint_path
    args_for_base_run_name = SimpleNamespace(**vars(args))
    args_for_base_run_name.checkpoint_path = ""
    base_run_name = get_run_name(args_for_base_run_name)
    run_name = base_run_name
    ft_run_name = get_run_name(args)
    checkpoint_dir = Path("./checkpoints")
    candidates = []

    if checkpoint_path:
        explicit_path = Path(checkpoint_path)
        candidates.append(explicit_path)
        if explicit_path.is_absolute() and len(explicit_path.parts) > 2:
            if explicit_path.parts[1] == "checkpoints":
                candidates.append(Path(".") / Path(*explicit_path.parts[1:]))

    for candidate_run_name in dict.fromkeys([base_run_name, ft_run_name]):
        candidates.extend(
            [
                checkpoint_dir / f"checkpoint_{candidate_run_name}_best.pt",
                checkpoint_dir / f"checkpoint_{candidate_run_name}_last.pt",
                checkpoint_dir / f"checkpoint_{candidate_run_name}.pt",
            ]
        )
        legacy_run_name = re.sub(r"_patchsize\d+(?=_stride)", "", candidate_run_name)
        if legacy_run_name != candidate_run_name:
            candidates.extend(
                [
                    checkpoint_dir / f"checkpoint_{legacy_run_name}_best.pt",
                    checkpoint_dir / f"checkpoint_{legacy_run_name}_last.pt",
                    checkpoint_dir / f"checkpoint_{legacy_run_name}.pt",
                ]
            )

    for candidate in candidates:
        if candidate.exists():
            if original_checkpoint_path and Path(original_checkpoint_path) != candidate:
                print(f"Using checkpoint fallback: {candidate}")
            return str(candidate), run_name
    raise FileNotFoundError(
        "No checkpoint found. Checked:\n" + "\n".join(str(path) for path in candidates)
    )


__all__ = [
    "cal_ssim_like_shanghai",
    "get_run_name",
    "load_checkpoint",
    "load_eval_obj",
    "load_train_objs",
    "prepare_dataloader",
    "resolve_eval_checkpoint_path",
    "save_metrics",
]
