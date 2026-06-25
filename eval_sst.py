"""
Evaluation for sea surface temperature (SST) data
-----------------------------------------
Follows eval.py (NSKT) evaluation flow:
* load model/checkpoint via utilities.load_train_objs
* deterministic multi-step evaluation
* per-step RFNE and Pearson correlation in original physical units
* SSIM in normalized z-scored SST space
"""

import os
import time
from datetime import datetime


def _restrict_to_first_visible_cuda_device():
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    visible_list = [device for device in visible_devices.split(",") if device]
    slurm_gpu_count = os.environ.get("SLURM_GPUS_ON_NODE", "")
    if os.getenv("USE_MULTI_GPU", "false").lower() in {"1", "true", "yes"}:
        if (
            visible_list
            and slurm_gpu_count.isdigit()
            and len(visible_list) > int(slurm_gpu_count)
        ):
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
                visible_list[: int(slurm_gpu_count)]
            )
        return
    if visible_list:
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_list[0]


_restrict_to_first_visible_cuda_device()

import cv2
import torch
import torch.nn as nn
import numpy as np
import scipy.stats
from torch.utils.data import DataLoader, Subset

from datasets.data_sea_temp import InputHandleEval
from utils.params_eval import get_args
from src.utilities import (
    cal_ssim_like_shanghai,
    load_train_objs,
    resolve_eval_checkpoint_path,
    save_metrics,
)


def inverse_z_score_sst(data, norm_stats, step):
    mean = norm_stats[:, step, 0][:, None, None, None]
    std = norm_stats[:, step, 1][:, None, None, None]
    return data * std + mean


class SSTEval(InputHandleEval):
    """
    Deterministic SST eval dataset:
    - condition_start: frame 0
    - condition_end: frame (total_interp_steps + 1)
    - targets: frames [1..total_interp_steps]
    """

    def __init__(self, data_path, total_interp_steps):
        input_param = {
            "path": data_path,
            "total_length": total_interp_steps + 2,  # start + targets + end
            "input_length": 2,
            "type": "test",
            "input_data_type": "float32",
        }
        super().__init__(input_param)
        self.total_interp_steps = total_interp_steps

    def __getitem__(self, index):
        if not (0 <= index < len(self.path_list)):
            raise IndexError(f"idx {index} out of range 0..{len(self.path_list)-1}")

        npy_paths = self.path_list[index]
        position = self.position_list[index]

        # Keep preprocessing consistent with data_sea_temp.py
        lat = self.latitude_map[npy_paths[0].split("/")[-4]][
            position[0] * 60 : position[0] * 60 + 128
        ]
        lon = self.longitude_map[npy_paths[0].split("/")[-4]][
            position[1] * 60 : position[1] * 60 + 128
        ]
        lat = np.array([lat[0] + (j * (lat[-1] - lat[0]) / 63.5) for j in range(64)])
        lon = np.array([lon[0] + (j * (lon[-1] - lon[0]) / 63.5) for j in range(64)])
        coor1, coor2 = np.meshgrid(lat, lon)
        coor = np.stack((coor1, coor2))
        _ = np.concatenate([np.sin(coor), np.cos(coor)], axis=0).astype(
            self.input_data_type
        ).transpose(1, 2, 0)

        time_list = []
        data = []
        norm_stats = []
        for i in range(self.length):
            npy_path = npy_paths[i]
            npy = np.load(npy_path)
            npy = npy[
                :,
                position[0] * 60 : position[0] * 60 + 256,
                position[1] * 60 : position[1] * 60 + 256,
            ]
            img = cv2.resize(npy[0], (128, 128))

            mean = np.mean(img)
            std = np.std(img)
            norm_stats.append((mean, std))
            data.append((img - mean) / std)

            cur_time = npy_path.split("/")[-1][:-4]
            cur_day = datetime.strptime(cur_time, "%Y%m%d")
            cur_day = cur_day.timetuple().tm_yday
            t = np.expand_dims(np.array([np.sin(cur_day / 366), np.cos(cur_day / 366)]), 1)
            time_list.append(t)

        data = np.stack(data, axis=0).astype(self.input_data_type)
        _ = np.concatenate(time_list, axis=1).astype(self.input_data_type)
        data = data.transpose((1, 2, 0))
        norm_stats = np.asarray(norm_stats, dtype=self.input_data_type)
        data = torch.from_numpy(data).float()
        norm_stats = torch.from_numpy(norm_stats).float()

        condition_start = data[..., 0].unsqueeze(0)
        condition_end = data[..., self.total_interp_steps + 1].unsqueeze(0)
        inputs = [condition_start, condition_end]

        targets = []
        for i in range(1, self.total_interp_steps + 1):
            targets.append(data[..., i].unsqueeze(0))

        cond_params = [
            torch.tensor(self.total_interp_steps, dtype=torch.float32),
            torch.tensor(0.0, dtype=torch.float32),
            norm_stats,
        ]
        return inputs, targets, cond_params


SST_SSIM_DATA_RANGE = 1.0


class SSTSampler(nn.Module):
    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.args = args

    def forward(
        self,
        condition_start,
        condition_end,
        reynolds_number,
        target_interp_step,
        total_interp_steps,
    ):
        if self.args.model in {"FLEX", "CrossFLEX"}:
            return self.model.sample(
                condition_start.shape[0],
                (1, self.args.patch_size, self.args.patch_size),
                condition_start,
                condition_end,
                reynolds_number,
                target_interp_step,
                total_interp_steps,
                condition_start.device,
            )
        return self.model.sample(
            condition_start,
            condition_end,
            reynolds_number,
            target_interp_step,
            total_interp_steps,
        )


if __name__ == "__main__":
    args = get_args()
    args.data_name = "sea_temp"

    print("Loading the trained model...")
    _, _, model, _, ema = load_train_objs(args=args)

    save_path, run_name = resolve_eval_checkpoint_path(args)
    print("Loading model from:", save_path)

    checkpoint = torch.load(save_path, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    if ema is not None and "ema" in checkpoint:
        ema.load_state_dict(checkpoint["ema"])

    seed = 0
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)

    model.to("cuda")
    model.eval()
    sampler = SSTSampler(model, args).to("cuda")
    use_multi_gpu = (
        os.getenv("USE_MULTI_GPU", "false").lower() in {"1", "true", "yes"}
        and torch.cuda.device_count() > 1
    )
    if use_multi_gpu:
        print(f"Using DataParallel SST eval on {torch.cuda.device_count()} GPUs")
        sampler = nn.DataParallel(sampler)
    else:
        print("Using single-GPU SST eval")
    sampler.eval()

    print("Loading the test dataset...")
    test_set = SSTEval(
        data_path=args.scratch_dir,
        total_interp_steps=args.total_interp_steps,
    )
    full_eval_size = len(test_set)
    # Reduce evaluation cost for large SST test sets.
    # Override with env var, e.g.:
    #   SST_EVAL_MAX_SAMPLES=5000 python eval_sst.py ...
    max_eval_samples = int(os.getenv("SST_EVAL_MAX_SAMPLES", "2048"))
    if max_eval_samples > 0 and len(test_set) > max_eval_samples:
        rng = np.random.default_rng(seed)
        subset_indices = rng.choice(
            len(test_set), size=max_eval_samples, replace=False
        )
        subset_indices = np.sort(subset_indices).tolist()
        test_set = Subset(test_set, subset_indices)
        print(f"Using SST eval subset: {len(subset_indices)}/{full_eval_size} samples")
    testloader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        pin_memory=True,
        shuffle=False,
        num_workers=8,
    )

    rfne_error = []
    r2s = []
    ssim_lst = []
    print(f"Number of batches: {len(testloader)}")

    if ema is not None:
        with ema.average_parameters():
            with torch.no_grad():
                model.eval()
                inference_time = []
                for i, (inputs, targets, cond_params) in enumerate(testloader):
                    print(i)
                    condition_start, condition_end = inputs
                    condition_start = condition_start.to("cuda")
                    condition_end = condition_end.to("cuda")

                    total_interp_steps, reynolds_number, norm_stats = cond_params
                    reynolds_number = reynolds_number.to("cuda")
                    total_interp_steps = total_interp_steps.to("cuda")
                    norm_stats = norm_stats.cpu().detach().numpy()

                    preds = []
                    start = time.time()
                    for ii in range(len(targets)):
                        target_interp_step = torch.tensor(
                            ii + 1, dtype=torch.float32, device="cuda"
                        )
                        target_interp_step = target_interp_step.repeat(condition_start.shape[0])

                        predictions = sampler(
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
                            target_original = inverse_z_score_sst(target, norm_stats, p + 1)
                            prediction_original = inverse_z_score_sst(
                                prediction, norm_stats, p + 1
                            )

                            error = (
                                np.linalg.norm(
                                    prediction_original[j, 0, :, :]
                                    - target_original[j, 0, :, :]
                                )
                                / np.linalg.norm(target_original[j, 0, :, :])
                            )
                            rfne_at_time.append(error)

                            cc = scipy.stats.pearsonr(
                                prediction_original[j, 0, :, :].flatten(),
                                target_original[j, 0, :, :].flatten(),
                            )[0]
                            cc_at_time.append(cc)

                            ssim = cal_ssim_like_shanghai(
                                prediction[j, 0, :, :],
                                target[j, 0, :, :],
                                data_range=SST_SSIM_DATA_RANGE,
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
                            "norm_stats": norm_stats,
                        }
                        os.makedirs("./samples", exist_ok=True)
                        sample_path = f"./samples/{run_name}_RESST_T{args.total_interp_steps}"
                        np.save(sample_path + ".npy", samples)
                        print("Generated samples saved...")
    else:
        with torch.no_grad():
            model.eval()
            inference_time = []
            for i, (inputs, targets, cond_params) in enumerate(testloader):
                print(i)
                condition_start, condition_end = inputs
                condition_start = condition_start.to("cuda")
                condition_end = condition_end.to("cuda")

                total_interp_steps, reynolds_number, norm_stats = cond_params
                reynolds_number = reynolds_number.to("cuda")
                total_interp_steps = total_interp_steps.to("cuda")
                norm_stats = norm_stats.cpu().detach().numpy()

                preds = []
                start = time.time()
                for ii in range(len(targets)):
                    target_interp_step = torch.tensor(
                        ii + 1, dtype=torch.float32, device="cuda"
                    )
                    target_interp_step = target_interp_step.repeat(condition_start.shape[0])

                    predictions = sampler(
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
                        target_original = inverse_z_score_sst(target, norm_stats, p + 1)
                        prediction_original = inverse_z_score_sst(
                            prediction, norm_stats, p + 1
                        )

                        error = (
                            np.linalg.norm(
                                prediction_original[j, 0, :, :]
                                - target_original[j, 0, :, :]
                            )
                            / np.linalg.norm(target_original[j, 0, :, :])
                        )
                        rfne_at_time.append(error)

                        cc = scipy.stats.pearsonr(
                            prediction_original[j, 0, :, :].flatten(),
                            target_original[j, 0, :, :].flatten(),
                        )[0]
                        cc_at_time.append(cc)

                        ssim = cal_ssim_like_shanghai(
                            prediction[j, 0, :, :],
                            target[j, 0, :, :],
                            data_range=SST_SSIM_DATA_RANGE,
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
                        "norm_stats": norm_stats,
                    }
                    os.makedirs("./samples", exist_ok=True)
                    sample_path = f"./samples/{run_name}_RESST_T{args.total_interp_steps}"
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
    metrics_save_path = f"./assets/eval_sst_{run_name}.txt"
    save_metrics(
        metrics=metrics,
        save_path=metrics_save_path,
        header=f"{run_name}_SST",
    )
