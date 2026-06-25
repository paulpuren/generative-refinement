import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import scipy.stats
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from forecasting.data_nskt import RE_EVAL_LIST
from forecasting.params import get_eval_args
from forecasting.utilities import (
    cal_ssim_like_shanghai,
    load_eval_obj,
    resolve_eval_checkpoint_path,
    save_metrics,
)
from src.utilities import materialize_checkpoint_modules


NSKT_SSIM_DATA_RANGE = 1.0


def _get_args_from_argv(argv):
    saved_argv = sys.argv
    try:
        sys.argv = argv
        return get_eval_args()
    finally:
        sys.argv = saved_argv


def load_yaml_config(config_path):
    if not config_path:
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("YAML config must be a key-value mapping.")
    return data


def get_eval_args_with_yaml():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default="")
    pre_args, remaining = pre_parser.parse_known_args()
    prog = sys.argv[0]
    default_args = _get_args_from_argv([prog])
    cli_args = _get_args_from_argv([prog] + remaining)
    merged = argparse.Namespace(**vars(default_args))
    config_data = load_yaml_config(pre_args.config)

    valid_keys = set(vars(default_args).keys())
    ignored_keys = sorted(set(config_data.keys()) - valid_keys)
    if ignored_keys:
        print("Ignoring config keys not used by forecasting eval:", ", ".join(ignored_keys))

    for key, value in config_data.items():
        if key in valid_keys:
            setattr(merged, key, value)
    for key, value in vars(cli_args).items():
        if value != getattr(default_args, key):
            setattr(merged, key, value)
    if "forecast_horizon" not in config_data and "total_interp_steps" in config_data:
        merged.forecast_horizon = config_data["total_interp_steps"]
    merged.total_interp_steps_train = 1
    merged.config = pre_args.config
    return merged


@torch.no_grad()
def autoregressive_rollout(model, condition_start, condition_end, reynolds_number, horizon, args):
    preds = []
    current = condition_start
    previous = condition_end
    batch_size = current.shape[0]
    target_interp_step = torch.ones(batch_size, dtype=torch.float32, device=current.device)
    total_interp_steps = torch.ones(batch_size, dtype=torch.float32, device=current.device)
    for _ in range(horizon):
        prediction = model.sample(
            batch_size,
            (1, args.patch_size, args.patch_size),
            current,
            previous,
            reynolds_number,
            target_interp_step,
            total_interp_steps,
            "cuda",
        )
        preds.append(prediction)
        previous = current
        current = prediction
    return preds


if __name__ == "__main__":
    args = get_eval_args_with_yaml()
    args.forecast_horizon = int(args.forecast_horizon)

    print("Loading forecasting model...")
    eval_set, model, ema = load_eval_obj(args=args)
    save_path, run_name = resolve_eval_checkpoint_path(args)
    print("Loading model from:", save_path)
    checkpoint = torch.load(save_path, weights_only=True)
    materialize_checkpoint_modules(model, checkpoint["model"])
    model.load_state_dict(checkpoint["model"])
    if ema is not None:
        ema.load_state_dict(checkpoint["ema"])

    seed = 0
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)

    model.to("cuda")
    model.eval()

    full_eval_size = len(eval_set)
    max_eval_samples = int(os.getenv("NSKT_EVAL_MAX_SAMPLES", "256"))
    if max_eval_samples > 0 and len(eval_set) > max_eval_samples:
        rng = np.random.default_rng(seed)
        subset_indices = rng.choice(len(eval_set), size=max_eval_samples, replace=False)
        subset_indices = np.sort(subset_indices).tolist()
        eval_set = Subset(eval_set, subset_indices)
        print(f"Using NSKT forecast eval subset: {len(subset_indices)}/{full_eval_size} samples")

    testloader = DataLoader(
        eval_set,
        batch_size=args.batch_size,
        pin_memory=True,
        shuffle=False,
        num_workers=8,
    )

    rfne_errors = []
    residual_rfne_errors = []
    persistence_rfne_errors = []
    r2s = []
    ssims = []
    inference_time = []
    print(f"Number of batches: {len(testloader)}")

    with ema.average_parameters():
        for i, (inputs, targets, cond_params) in enumerate(testloader):
            print(i)
            condition_start, condition_end = inputs
            condition_start = condition_start.to("cuda")
            condition_end = condition_end.to("cuda")
            _, reynolds_number = cond_params
            reynolds_number = reynolds_number.to("cuda")

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            preds_torch = autoregressive_rollout(
                model,
                condition_start,
                condition_end,
                reynolds_number,
                len(targets),
                args,
            )
            end_event.record()
            end_event.synchronize()
            inference_time.append(start_event.elapsed_time(end_event) / 1000.0 / len(targets))

            preds = [pred.cpu().numpy() for pred in preds_torch]
            condition_start_np = condition_start.cpu().numpy()

            for j in range(condition_start.shape[0]):
                rfne_at_steps = []
                residual_rfne_at_steps = []
                persistence_rfne_at_steps = []
                r2_at_steps = []
                ssim_at_steps = []
                for step_id, target_torch in enumerate(targets):
                    target = target_torch.numpy()
                    prediction = preds[step_id]
                    denom = max(np.linalg.norm(target[j, 0]), 1e-12)
                    rfne_at_steps.append(
                        np.linalg.norm(prediction[j, 0] - target[j, 0]) / denom
                    )
                    residual_denom = max(
                        np.linalg.norm(target[j, 0] - condition_start_np[j, 0]),
                        1e-12,
                    )
                    residual_rfne_at_steps.append(
                        np.linalg.norm(prediction[j, 0] - target[j, 0]) / residual_denom
                    )
                    persistence_rfne_at_steps.append(
                        np.linalg.norm(condition_start_np[j, 0] - target[j, 0]) / denom
                    )
                    r2_at_steps.append(
                        scipy.stats.pearsonr(
                            prediction[j, 0].flatten(),
                            target[j, 0].flatten(),
                        )[0]
                    )
                    ssim_at_steps.append(
                        cal_ssim_like_shanghai(
                            prediction[j, 0],
                            target[j, 0],
                            data_range=NSKT_SSIM_DATA_RANGE,
                        )
                    )
                rfne_errors.append(rfne_at_steps)
                residual_rfne_errors.append(residual_rfne_at_steps)
                persistence_rfne_errors.append(persistence_rfne_at_steps)
                r2s.append(r2_at_steps)
                ssims.append(ssim_at_steps)

            print(np.mean(np.vstack(r2s), axis=0))

            if i == 0:
                os.makedirs("./samples", exist_ok=True)
                re_tag = "all" if args.re_id < 0 else RE_EVAL_LIST[args.re_id]
                sample_path = f"./samples/{run_name}_forecast_RE{re_tag}_H{args.forecast_horizon}.npy"
                np.save(
                    sample_path,
                    {
                        "conditioning_snapshots": condition_start.cpu().numpy(),
                        "targets": [target.numpy() for target in targets],
                        "predictions": preds,
                    },
                )
                print("Generated forecast samples saved...")

    avg_rfne = np.mean(np.vstack(rfne_errors), axis=0)
    avg_residual_rfne = np.mean(np.vstack(residual_rfne_errors), axis=0)
    avg_persistence_rfne = np.mean(np.vstack(persistence_rfne_errors), axis=0)
    avg_r2 = np.mean(np.vstack(r2s), axis=0)
    avg_ssim = np.mean(np.vstack(ssims), axis=0)
    avg_infer_time = np.mean(inference_time, axis=0)

    print(f"Average RFNE={repr(avg_rfne)}")
    print(f"Average residual-normalized RFNE={repr(avg_residual_rfne)}")
    print(f"Average persistence baseline RFNE={repr(avg_persistence_rfne)}")
    print(f"Average Pearson correlation coefficients={repr(avg_r2)}")
    print(f"Average SSIM value={repr(avg_ssim)}")
    print(f"Average Inference Time={repr(avg_infer_time)}")

    metrics = {
        "avg_rfne_steps": avg_rfne,
        "avg_residual_rfne_steps": avg_residual_rfne,
        "avg_persistence_rfne_steps": avg_persistence_rfne,
        "avg_r2_steps": avg_r2,
        "avg_ssim_steps": avg_ssim,
        "avg_rfne_value": np.mean(avg_rfne, axis=0),
        "avg_residual_rfne_value": np.mean(avg_residual_rfne, axis=0),
        "avg_persistence_rfne_value": np.mean(avg_persistence_rfne, axis=0),
        "avg_r2_value": np.mean(avg_r2, axis=0),
        "avg_ssim_value": np.mean(avg_ssim, axis=0),
        "avg_run_time": avg_infer_time,
    }
    os.makedirs("./assets", exist_ok=True)
    re_tag = "all" if args.re_id < 0 else RE_EVAL_LIST[args.re_id]
    metrics_save_path = f"./assets/eval_forecast_re{re_tag}_{run_name}.txt"
    save_metrics(metrics=metrics, save_path=metrics_save_path, header=f"{run_name}_RE{re_tag}")
