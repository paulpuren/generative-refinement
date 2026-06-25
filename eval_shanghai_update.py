"""
Evaluation for Shanghai radar data (updated protocol)
-----------------------------------------------------
Follows eval.py (NSKT) evaluation flow:
* load model/checkpoint via utilities.load_train_objs
* deterministic multi-step evaluation
* per-step RFNE and Pearson correlation
* Shanghai SSIM metric
"""

import os
import time
import sys
import argparse
import h5py
import yaml
import torch
import numpy as np
import scipy.stats
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from utils.params_eval import get_args
from src.utilities import (
    cal_ssim_like_shanghai,
    load_eval_obj,
    resolve_eval_checkpoint_path,
    save_metrics,
)


PIXEL_SCALE = 90.0


def materialize_checkpoint_modules(model, state_dict):
    latent_fusion_keys = {
        key.split(".")[1]
        for key in state_dict
        if key.startswith("latent_fusion.")
    }
    feature_fusion_keys = {
        key.split(".")[1]
        for key in state_dict
        if key.startswith("feature_fusion.")
    }

    if latent_fusion_keys and hasattr(model, "_get_latent_fusion"):
        for channels_key in sorted(latent_fusion_keys, key=int):
            model._get_latent_fusion(
                torch.empty(1, int(channels_key), 1, 1, device="cpu")
            )

    if feature_fusion_keys and hasattr(model, "_get_feature_fusion"):
        for channels_key in sorted(feature_fusion_keys, key=int):
            model._get_feature_fusion(
                torch.empty(1, int(channels_key), 1, 1, device="cpu")
            )


def _get_args_from_argv(argv):
    saved_argv = sys.argv
    try:
        sys.argv = argv
        return get_args()
    finally:
        sys.argv = saved_argv


def load_yaml_config(config_path):
    if not config_path:
        return {}
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
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
        print(
            "Ignoring config keys not used by eval:",
            ", ".join(ignored_keys),
        )

    for key, value in config_data.items():
        if key in valid_keys:
            setattr(merged, key, value)

    for key, value in vars(cli_args).items():
        if value != getattr(default_args, key):
            setattr(merged, key, value)

    if (
        getattr(merged, "total_interp_steps", None) == default_args.total_interp_steps
        and "total_interp_steps" not in config_data
        and "total_interp_steps_train" in config_data
    ):
        merged.total_interp_steps = config_data["total_interp_steps_train"]

    setattr(merged, "config", pre_args.config)
    return merged

class ShanghaiEval(Dataset):
    def __init__(
            self, 
            total_interp_steps, 
            data_path, 
            img_size, 
            split="test", 
            trans=None
        ):
        super().__init__()
        self.total_interp_steps = total_interp_steps
        self.data_path = data_path
        self.img_size = img_size
        self.split = split if split != "val" else "test"

        with h5py.File(data_path, "r") as f:
            self.all_len = int(f[self.split]["all_len"][()])

        self.transform = trans or transforms.Compose(
            [transforms.Resize((img_size, img_size))]
        )

    def __len__(self):
        return self.all_len

    def __getitem__(self, index):
        with h5py.File(self.data_path, "r") as f:
            imgs = f[self.split][str(index)][()]
            frames = torch.from_numpy(imgs).float().squeeze()
            frames = frames / 255.0
            frames = self.transform(frames)

        condition_start = frames[0].unsqueeze(0)
        condition_end = frames[self.total_interp_steps + 1].unsqueeze(0)
        inputs = [condition_start, condition_end]

        targets = []
        for i in range(1, self.total_interp_steps + 1):
            targets.append(frames[i].unsqueeze(0))

        cond_params = [
            torch.tensor(self.total_interp_steps, dtype=torch.float32),
            torch.tensor(0.0, dtype=torch.float32),
        ]
        return inputs, targets, cond_params


if __name__ == "__main__":
    args = get_eval_args_with_yaml()
    args.data_name = "shanghai"

    print("Loading the trained model...")
    _, model, ema = load_eval_obj(args=args)

    save_path, run_name = resolve_eval_checkpoint_path(args)
    print("Loading model from:", save_path)
    checkpoint = torch.load(save_path, weights_only=True)
    materialize_checkpoint_modules(model, checkpoint["model"])
    model.load_state_dict(checkpoint["model"])
    if ema is not None and "ema" in checkpoint:
        ema.load_state_dict(checkpoint["ema"])

    seed = 0
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)

    model.to("cuda")
    model.eval()

    print("Loading the test dataset...")
    test_set = ShanghaiEval(
        total_interp_steps=args.total_interp_steps,
        data_path=args.scratch_dir,
        img_size=args.patch_size,
        split="test",
    )
    testloader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        pin_memory=True,
        shuffle=False,
        num_workers=8,
    )

    print(f"Number of batches: {len(testloader)}")

    rfne_error = []
    r2s = []
    ssim_lst = []
    inference_time = []

    context = ema.average_parameters() if ema is not None else torch.no_grad()
    if ema is None:
        # torch.no_grad() will be entered below; keep this branch explicit for readability.
        pass

    with context:
        with torch.no_grad():
            model.eval()
            for i, (inputs, targets, cond_params) in enumerate(testloader):
                print(i)
                condition_start, condition_end = inputs
                condition_start = condition_start.to("cuda")
                condition_end = condition_end.to("cuda")

                total_interp_steps, reynolds_number = cond_params
                reynolds_number = reynolds_number.to("cuda")
                total_interp_steps = total_interp_steps.to("cuda")

                preds = []
                start = time.time()
                for ii in range(len(targets)):
                    target_interp_step = torch.tensor(
                        ii + 1,
                        dtype=torch.float32,
                        device="cuda",
                    ).repeat(condition_start.shape[0])

                    if ema is not None:
                        predictions = model.sample(
                            condition_start.shape[0],
                            (1, args.patch_size, args.patch_size),
                            condition_start,
                            condition_end,
                            reynolds_number,
                            target_interp_step,
                            total_interp_steps,
                            "cuda",
                        )
                    else:
                        predictions = model.sample(
                            condition_start,
                            condition_end,
                            reynolds_number,
                            target_interp_step,
                            total_interp_steps,
                        )
                    preds.append(predictions.cpu().detach().numpy())
                end = time.time()
                inference_time.append((end - start) / len(targets))

                for j in range(predictions.shape[0]):
                    rfne_at_time = []
                    cc_at_time = []
                    ssim_at_time = []
                    for p in range(len(targets)):
                        target = targets[p].cpu().detach().numpy()
                        prediction = preds[p]

                        error = np.linalg.norm(
                            prediction[j, 0, :, :] - target[j, 0, :, :]
                        ) / np.linalg.norm(target[j, 0, :, :])
                        rfne_at_time.append(error)

                        cc = scipy.stats.pearsonr(
                            prediction[j, 0, :, :].flatten(),
                            target[j, 0, :, :].flatten(),
                        )[0]
                        cc_at_time.append(cc)

                        ssim = cal_ssim_like_shanghai(
                            prediction[j, 0, :, :] * PIXEL_SCALE,
                            target[j, 0, :, :] * PIXEL_SCALE,
                            data_range=PIXEL_SCALE,
                        )
                        ssim_at_time.append(ssim)

                    rfne_error.append(rfne_at_time)
                    r2s.append(cc_at_time)
                    ssim_lst.append(ssim_at_time)

                print(np.mean(np.vstack(r2s), axis=0))

                if i == 0:
                    samples = {
                        "conditioning_snapshots": condition_start.cpu().detach().numpy(),
                        "targets": targets,
                        "predictions": preds,
                    }
                    os.makedirs("./samples", exist_ok=True)
                    sample_path = f"./samples/{run_name}_Shanghai_T{args.total_interp_steps}"
                    np.save(sample_path + ".npy", samples)
                    print("Generated samples saved...")

    avg_rfne = np.mean(np.vstack(rfne_error), axis=0)
    avg_r2 = np.mean(np.vstack(r2s), axis=0)
    avg_ssim = np.mean(np.vstack(ssim_lst), axis=0)
    avg_infer_time = np.mean(inference_time, axis=0)

    print(f"Average RFNE={repr(avg_rfne)}")
    print(f"Average Pearson correlation coefficients={repr(avg_r2)}")
    print(f"Average SSIM value={repr(avg_ssim)}")
    print(f"Average Inference Time={repr(avg_infer_time)}")

    metrics = {
        "avg_rfne_steps": avg_rfne,
        "avg_r2_steps": avg_r2,
        "avg_ssim_steps": avg_ssim,
        "avg_rfne_value": np.mean(avg_rfne, axis=0),
        "avg_r2_value": np.mean(avg_r2, axis=0),
        "avg_ssim_value": np.mean(avg_ssim, axis=0),
        "avg_run_time": avg_infer_time,
    }

    metrics_save_path = f"./assets/eval_shanghai_{run_name}.txt"
    save_metrics(
        metrics=metrics,
        save_path=metrics_save_path,
        header=f"{run_name}_Shanghai",
    )
