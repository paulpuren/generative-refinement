#!/usr/bin/env python3
"""GPU-only per-frame inference timing for interpolation models.

This intentionally does no metric computation and does not copy predictions
back to CPU. It measures only the CUDA work inside each model's forward/sample
call by using CUDA events.
"""

import argparse
import csv
import os
import sys
try:
    from contextlib import nullcontext
except ImportError:
    class nullcontext:
        def __init__(self, enter_result=None):
            self.enter_result = enter_result

        def __enter__(self):
            return self.enter_result

        def __exit__(self, *excinfo):
            return False
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_yaml_config(config_path):
    if not config_path:
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return data


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark GPU-only per-frame inference time."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["UNet", "FLEX", "FLEXResidual", "SuperSloMo", "EDEN"],
    )
    parser.add_argument("--data_name", choices=["nskt", "shanghai", "sea_temp", "sst"])
    parser.add_argument("--config", default="")
    parser.add_argument("--checkpoint_path", default="")
    parser.add_argument("--vae_checkpoint_path", default="")
    parser.add_argument("--dit_checkpoint_path", default="")
    parser.add_argument("--scratch_dir", default="")
    parser.add_argument("--data_path", default="")
    parser.add_argument("--re_id", type=int)
    parser.add_argument("--total_interp_steps", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--patch_size", type=int)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--crop_size", type=int)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_batches", type=int, default=20)
    parser.add_argument("--warmup_batches", type=int, default=2)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--output_csv", default="assets/inference_time_per_frame.csv")
    parser.add_argument("--seed", type=int, default=0)
    args, extra_args = parser.parse_known_args()
    return args, extra_args


def cuda_elapsed_seconds(fn):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    result = fn()
    end_event.record()
    end_event.synchronize()
    return start_event.elapsed_time(end_event) / 1000.0, result


def maybe_subset(dataset, max_samples, seed):
    if max_samples <= 0 or len(dataset) <= max_samples:
        return dataset
    from torch.utils.data import Subset

    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(dataset), size=max_samples, replace=False)).tolist()
    return Subset(dataset, indices)


def write_result(path, row):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists()
    with output_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def root_args_from_config(cli_args, extra_args):
    from utils.params_eval import get_args

    saved_argv = sys.argv
    try:
        sys.argv = [sys.argv[0]]
        default_args = get_args()
        sys.argv = [sys.argv[0]] + extra_args
        parsed_extra = get_args()
    finally:
        sys.argv = saved_argv

    merged = argparse.Namespace(**vars(default_args))
    config_data = load_yaml_config(cli_args.config)
    valid_keys = set(vars(default_args))

    for key, value in config_data.items():
        if key in valid_keys:
            setattr(merged, key, value)

    for key, value in vars(parsed_extra).items():
        if value != getattr(default_args, key):
            setattr(merged, key, value)

    explicit = {
        "model": cli_args.model,
        "data_name": cli_args.data_name,
        "checkpoint_path": cli_args.checkpoint_path,
        "scratch_dir": cli_args.scratch_dir or cli_args.data_path,
        "re_id": cli_args.re_id,
        "total_interp_steps": cli_args.total_interp_steps,
        "batch_size": cli_args.batch_size,
        "patch_size": cli_args.patch_size,
        "stride": cli_args.stride,
        "crop_size": cli_args.crop_size,
    }
    for key, value in explicit.items():
        if value not in (None, ""):
            setattr(merged, key, value)

    if (
        getattr(merged, "total_interp_steps", None) == default_args.total_interp_steps
        and "total_interp_steps" not in config_data
        and "total_interp_steps_train" in config_data
    ):
        merged.total_interp_steps = config_data["total_interp_steps_train"]

    if merged.data_name == "sst":
        merged.data_name = "sea_temp"
    return merged


def build_fixed_root_dataset(args):
    from datasets.data_nskt_updated import NSKT_eval
    from eval_super_slomo import ShanghaiEval, SSTEval

    if args.data_name == "nskt":
        dataset = NSKT_eval(
            patch_size=args.patch_size,
            crop_size=args.crop_size,
            stride=args.stride,
            num_interp_steps=args.total_interp_steps,
            re_id=args.re_id,
            scratch_dir=args.scratch_dir,
        )
    elif args.data_name == "shanghai":
        dataset = ShanghaiEval(
            total_interp_steps=args.total_interp_steps,
            data_path=args.scratch_dir,
            img_size=args.patch_size,
            split="test",
        )
    elif args.data_name == "sea_temp":
        dataset = SSTEval(
            data_path=args.scratch_dir,
            total_interp_steps=args.total_interp_steps,
        )
    else:
        raise ValueError(f"Unsupported data_name: {args.data_name}")
    return dataset


def load_root_model(args):
    import torch
    from torch_ema import ExponentialMovingAverage

    from src.ablation.diffusion_model_residual import DiffusionModel as DiffusionModelResidual
    from src.diffusion_model import DiffusionModel
    from src.flex import FLEX
    from src.unet import UNet
    from src.utilities import resolve_eval_checkpoint_path

    ema = None
    if args.model in {"FLEX", "FLEXResidual"}:
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
        diffusion_cls = DiffusionModelResidual if args.model == "FLEXResidual" else DiffusionModel
        model = diffusion_cls(
            encoder=encoder.cuda(),
            decoder=decoder.cuda(),
            task_encoder=task_encoder.cuda(),
            task_encoder_end=task_encoder_end.cuda() if task_encoder_end is not None else None,
            diff_steps=args.time_steps,
            prediction_type=args.prediction_type,
            criterion=torch.nn.L1Loss(),
            dt_normalization_scale=float(args.total_interp_steps),
            condition_on_re=args.condition_on_re,
            condition_on_total_interp_steps=args.condition_on_total_interp_steps,
        )
        ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
    elif args.model == "UNet":
        model = UNet(
            image_size=args.patch_size,
            in_channels=2,
            out_channels=1,
            base_width=args.base_width,
        )
    else:
        raise ValueError(f"Unsupported root model: {args.model}")

    save_path, run_name = resolve_eval_checkpoint_path(args)
    checkpoint = torch.load(save_path, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    if ema is not None and "ema" in checkpoint:
        ema.load_state_dict(checkpoint["ema"])
    model.to("cuda")
    model.eval()
    return model, ema, save_path, run_name


def load_super_slomo_model(args):
    import torch
    import src.super_slomo as slomo_model
    from eval_super_slomo import run_super_slomo_step
    from src.utilities import resolve_eval_checkpoint_path

    slomo_model.configure_time_grid(args.data_name, args.total_interp_steps)
    flow_comp = slomo_model.UNet(2, 4)
    flow_interp = slomo_model.UNet(12, 5)
    back_warp = slomo_model.backWarp(args.patch_size, args.patch_size, "cuda")

    save_path, run_name = resolve_eval_checkpoint_path(args)
    checkpoint = torch.load(save_path, weights_only=True)
    flow_comp.load_state_dict(checkpoint["flowComp"])
    flow_interp.load_state_dict(checkpoint["ArbTimeFlowIntrp"])

    flow_comp = flow_comp.to("cuda").eval()
    flow_interp = flow_interp.to("cuda").eval()
    back_warp = back_warp.to("cuda")
    return (flow_comp, flow_interp, back_warp, run_super_slomo_step), save_path, run_name


def time_root_batch(model, args, batch):
    inputs, targets, cond_params = batch
    condition_start, condition_end = inputs
    condition_start = condition_start.to("cuda", non_blocking=True)
    condition_end = condition_end.to("cuda", non_blocking=True)
    total_interp_steps = cond_params[0].to("cuda", non_blocking=True)
    fluid_condition = cond_params[1].to("cuda", non_blocking=True)

    elapsed = 0.0
    batch_size = condition_start.shape[0]
    for step in range(len(targets)):
        target_interp_step = torch.full(
            (batch_size,), step + 1, dtype=torch.float32, device="cuda"
        )

        if args.model in {"FLEX", "FLEXResidual"}:
            def infer():
                return model.sample(
                    batch_size,
                    (1, args.patch_size, args.patch_size),
                    condition_start,
                    condition_end,
                    fluid_condition,
                    target_interp_step,
                    total_interp_steps,
                    "cuda",
                )
        else:
            def infer():
                return model.sample(
                    condition_start,
                    condition_end,
                    fluid_condition,
                    target_interp_step,
                    total_interp_steps,
                )

        step_elapsed, prediction = cuda_elapsed_seconds(infer)
        elapsed += step_elapsed
        del prediction
    return elapsed / len(targets)


def time_super_slomo_batch(bundle, batch):
    flow_comp, flow_interp, back_warp, run_super_slomo_step = bundle
    inputs, targets, _ = batch
    i0, i1 = inputs
    i0 = i0.to("cuda", non_blocking=True)
    i1 = i1.to("cuda", non_blocking=True)

    elapsed = 0.0
    for step in range(len(targets)):
        target_interp_step = torch.full(
            (i0.shape[0],), step + 1, dtype=torch.float32, device="cuda"
        )

        def infer():
            return run_super_slomo_step(
                flow_comp, flow_interp, back_warp, i0, i1, target_interp_step, "cuda"
            )

        step_elapsed, prediction = cuda_elapsed_seconds(infer)
        elapsed += step_elapsed
        del prediction
    return elapsed / len(targets)


def benchmark_root(cli_args, extra_args):
    from torch.utils.data import DataLoader

    args = root_args_from_config(cli_args, extra_args)
    dataset = maybe_subset(build_fixed_root_dataset(args), cli_args.max_samples, cli_args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=cli_args.num_workers,
    )

    if args.model == "SuperSloMo":
        model_obj, checkpoint_path, run_name = load_super_slomo_model(args)
        time_batch = lambda batch: time_super_slomo_batch(model_obj, batch)
        ema_context = nullcontext()
    else:
        model, ema, checkpoint_path, run_name = load_root_model(args)
        time_batch = lambda batch: time_root_batch(model, args, batch)
        ema_context = ema.average_parameters() if ema is not None else nullcontext()

    times = []
    with torch.no_grad(), ema_context:
        for batch_idx, batch in enumerate(loader):
            batch_time = time_batch(batch)
            if batch_idx >= cli_args.warmup_batches:
                times.append(batch_time)
            if batch_idx + 1 >= cli_args.warmup_batches + cli_args.num_batches:
                break

    return args, run_name, str(checkpoint_path), np.asarray(times, dtype=float)


def eden_args_from_config(cli_args):
    config_data = load_yaml_config(cli_args.config)
    args = SimpleNamespace(**config_data)
    explicit = {
        "dataset_name": cli_args.data_name,
        "vae_checkpoint_path": cli_args.vae_checkpoint_path,
        "dit_checkpoint_path": cli_args.dit_checkpoint_path,
        "data_path": cli_args.data_path or cli_args.scratch_dir,
        "scratch_dir": cli_args.scratch_dir or cli_args.data_path,
        "re_id": cli_args.re_id,
        "total_interp_steps": cli_args.total_interp_steps,
        "batch_size": cli_args.batch_size,
        "patch_size": cli_args.patch_size,
        "stride": cli_args.stride,
        "crop_size": cli_args.crop_size,
    }
    if explicit["dataset_name"] == "sea_temp":
        explicit["dataset_name"] = "sst"
    for key, value in explicit.items():
        if value not in (None, ""):
            setattr(args, key, value)
    if not getattr(args, "data_path", None) and getattr(args, "scratch_dir", None):
        args.data_path = args.scratch_dir
    return args


def benchmark_eden(cli_args):
    from torch.utils.data import DataLoader

    eden_dir = REPO_ROOT / "EDEN"
    sys.path.insert(0, str(eden_dir))
    old_cwd = Path.cwd()

    def resolve_before_chdir(path_value):
        if not path_value:
            return path_value
        path = Path(path_value)
        if path.is_absolute():
            return str(path)
        candidates = [
            old_cwd / path,
            eden_dir / path,
            REPO_ROOT / path,
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate.resolve())
        return str((old_cwd / path).resolve())

    cli_args.config = resolve_before_chdir(cli_args.config)
    cli_args.vae_checkpoint_path = resolve_before_chdir(cli_args.vae_checkpoint_path)
    cli_args.dit_checkpoint_path = resolve_before_chdir(cli_args.dit_checkpoint_path)
    os.chdir(eden_dir)
    try:
        from eval_scientific import build_eval_dataset, sample_with_eden
        from src.models import load_model
        from src.transport import Sampler, create_transport

        args = eden_args_from_config(cli_args)
        dataset = maybe_subset(build_eval_dataset(args), cli_args.max_samples, cli_args.seed)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=cli_args.num_workers,
        )

        dit = load_model("EDEN_DiT", **args.model_args).to("cuda")
        vae = load_model("EDEN_VAE", **args.vae_args).to("cuda")
        dit_ckpt = torch.load(args.dit_checkpoint_path, map_location="cpu")
        vae_ckpt = torch.load(args.vae_checkpoint_path, map_location="cpu")
        dit.load_state_dict(dit_ckpt["eden_dit"])
        vae.load_state_dict(vae_ckpt["eden_vae"], strict=False)
        dit.eval()
        vae.eval()

        transport = create_transport(**args.transport)
        sampler = Sampler(transport)
        sample_fn = sampler.sample_ode(
            sampling_method=args.sampling_method,
            num_steps=args.sample_steps,
            atol=args.atol,
            rtol=args.rtol,
        )

        times = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                inputs, targets, cond_params = batch
                condition_start, condition_end = inputs
                condition_start = condition_start.to("cuda", non_blocking=True)
                condition_end = condition_end.to("cuda", non_blocking=True)
                total_interp_steps = cond_params[0].to("cuda", non_blocking=True)

                elapsed = 0.0
                for step in range(len(targets)):
                    interp = (
                        torch.ones_like(total_interp_steps, dtype=torch.float32)
                        * (step + 1)
                    ) / (total_interp_steps.float() + 1.0)

                    def infer():
                        return sample_with_eden(
                            dit=dit,
                            vae=vae,
                            sample_fn=sample_fn,
                            cond_start=condition_start,
                            cond_end=condition_end,
                            interp=interp,
                            args=args,
                            device="cuda",
                        )

                    step_elapsed, prediction = cuda_elapsed_seconds(infer)
                    elapsed += step_elapsed
                    del prediction

                batch_time = elapsed / len(targets)
                if batch_idx >= cli_args.warmup_batches:
                    times.append(batch_time)
                if batch_idx + 1 >= cli_args.warmup_batches + cli_args.num_batches:
                    break

        run_name = getattr(args, "run_name", "EDEN")
        checkpoint_path = f"vae={args.vae_checkpoint_path};dit={args.dit_checkpoint_path}"
        return args, run_name, checkpoint_path, np.asarray(times, dtype=float)
    finally:
        os.chdir(old_cwd)


def main():
    global torch
    cli_args, extra_args = parse_args()
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GPU-only inference timing.")

    np.random.seed(cli_args.seed)
    torch.manual_seed(cli_args.seed)

    if cli_args.model == "EDEN":
        args, run_name, checkpoint_path, times = benchmark_eden(cli_args)
        data_name = getattr(args, "dataset_name", cli_args.data_name)
        batch_size = args.batch_size
        total_interp_steps = args.total_interp_steps
    else:
        args, run_name, checkpoint_path, times = benchmark_root(cli_args, extra_args)
        data_name = args.data_name
        batch_size = args.batch_size
        total_interp_steps = args.total_interp_steps

    if times.size == 0:
        raise RuntimeError("No timed batches were collected.")

    mean_seconds_per_batch_frame = float(np.mean(times))
    std_seconds_per_batch_frame = float(np.std(times))
    median_seconds_per_batch_frame = float(np.median(times))
    mean_seconds_per_sample_frame = mean_seconds_per_batch_frame / float(batch_size)
    std_seconds_per_sample_frame = std_seconds_per_batch_frame / float(batch_size)
    median_seconds_per_sample_frame = median_seconds_per_batch_frame / float(batch_size)

    row = {
        "model": cli_args.model,
        "data_name": data_name,
        "run_name": run_name,
        "checkpoint_path": checkpoint_path,
        "total_interp_steps": total_interp_steps,
        "batch_size": batch_size,
        "warmup_batches": cli_args.warmup_batches,
        "timed_batches": int(times.size),
        # Backward-compatible name. This is per generated interpolation step
        # for the whole batch, not normalized by batch size.
        "mean_seconds_per_frame": mean_seconds_per_batch_frame,
        "std_seconds_per_frame": std_seconds_per_batch_frame,
        "median_seconds_per_frame": median_seconds_per_batch_frame,
        "mean_seconds_per_batch_frame": mean_seconds_per_batch_frame,
        "std_seconds_per_batch_frame": std_seconds_per_batch_frame,
        "median_seconds_per_batch_frame": median_seconds_per_batch_frame,
        "mean_seconds_per_sample_frame": mean_seconds_per_sample_frame,
        "std_seconds_per_sample_frame": std_seconds_per_sample_frame,
        "median_seconds_per_sample_frame": median_seconds_per_sample_frame,
    }

    print(row)
    write_result(cli_args.output_csv, row)
    print(f"Appended timing result to {cli_args.output_csv}")


if __name__ == "__main__":
    main()
