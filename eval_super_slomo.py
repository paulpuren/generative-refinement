"""
Evaluation for SuperSloMo on NSKT, Shanghai, and SST.
"""

import argparse
import os
import sys
from datetime import datetime


def _restrict_to_first_visible_cuda_device():
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if "," in visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices.split(",", 1)[0]


_restrict_to_first_visible_cuda_device()

import cv2
import h5py
import numpy as np
import scipy.stats
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

import src.super_slomo as slomo_model
from datasets.data_nskt import NSKT_eval
from datasets.data_sea_temp import InputHandleEval
from utils.params_eval import get_args
from src.utilities import cal_ssim_like_shanghai, resolve_eval_checkpoint_path, save_metrics


RE_EVAL_LIST = [
    600, 1000, 2000, 4000, 8000,
    12000, 16000, 24000, 32000, 36000,
]
DATA_RANGE_BY_DATASET = {
    "nskt": 1.0,
    "shanghai": 1.0,
    "sea_temp": 1.0,
}


def inverse_z_score_sst(data, norm_stats, step):
    mean = norm_stats[:, step, 0][:, None, None, None]
    std = norm_stats[:, step, 1][:, None, None, None]
    return data * std + mean


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
        print("Ignoring config keys not used by eval:", ", ".join(ignored_keys))

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
    def __init__(self, total_interp_steps, data_path, img_size, split="test", trans=None):
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


class SSTEval(InputHandleEval):
    def __init__(self, data_path, total_interp_steps):
        input_param = {
            "path": data_path,
            "total_length": total_interp_steps + 2,
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


def build_eval_dataset(args):
    data_name = slomo_model.normalize_dataset_name(args.data_name)
    if data_name == "nskt":
        dataset = NSKT_eval(
            patch_size=args.patch_size,
            stride=args.stride,
            num_interp_steps=args.total_interp_steps,
            re_id=args.re_id,
            scratch_dir=args.scratch_dir,
        )
    elif data_name == "shanghai":
        dataset = ShanghaiEval(
            total_interp_steps=args.total_interp_steps,
            data_path=args.scratch_dir,
            img_size=args.patch_size,
            split="test",
        )
    elif data_name == "sea_temp":
        dataset = SSTEval(
            data_path=args.scratch_dir,
            total_interp_steps=args.total_interp_steps,
        )
        max_eval_samples = int(os.getenv("SST_EVAL_MAX_SAMPLES", "2048"))
        if max_eval_samples > 0 and len(dataset) > max_eval_samples:
            rng = np.random.default_rng(0)
            subset_indices = rng.choice(len(dataset), size=max_eval_samples, replace=False)
            subset_indices = np.sort(subset_indices).tolist()
            dataset = Subset(dataset, subset_indices)
            print(f"Using SST eval subset: {len(subset_indices)} samples")
    else:
        raise ValueError(f"Unsupported dataset: {args.data_name}")
    return dataset, data_name


def run_super_slomo_step(flow_comp, flow_interp, back_warp, I0, I1, target_interp_step, device):
    flow_out = flow_comp(torch.cat((I0, I1), dim=1))
    F_0_1 = flow_out[:, :2, :, :]
    F_1_0 = flow_out[:, 2:, :, :]

    f_coeff = slomo_model.getFlowCoeff(target_interp_step, device)
    F_t_0 = f_coeff[0] * F_0_1 + f_coeff[1] * F_1_0
    F_t_1 = f_coeff[2] * F_0_1 + f_coeff[3] * F_1_0

    g_I0_F_t_0 = back_warp(I0, F_t_0)
    g_I1_F_t_1 = back_warp(I1, F_t_1)

    intrp_out = flow_interp(
        torch.cat(
            (I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0),
            dim=1,
        )
    )

    F_t_0_f = intrp_out[:, :2, :, :] + F_t_0
    F_t_1_f = intrp_out[:, 2:4, :, :] + F_t_1
    V_t_0 = F.sigmoid(intrp_out[:, 4:5, :, :])
    V_t_1 = 1 - V_t_0

    g_I0_F_t_0_f = back_warp(I0, F_t_0_f)
    g_I1_F_t_1_f = back_warp(I1, F_t_1_f)

    w_coeff = slomo_model.getWarpCoeff(target_interp_step, device)
    prediction = (
        w_coeff[0] * V_t_0 * g_I0_F_t_0_f + w_coeff[1] * V_t_1 * g_I1_F_t_1_f
    ) / (w_coeff[0] * V_t_0 + w_coeff[1] * V_t_1)
    return prediction


def metrics_output_paths(args, run_name, dataset_name):
    if dataset_name == "nskt":
        re_value = RE_EVAL_LIST[args.re_id]
        sample_path = f"./samples/{run_name}_RE{re_value}_T{args.total_interp_steps}"
        metrics_path = f"./assets/eval_re{re_value}_{run_name}.txt"
        header = f"{run_name}_RE{re_value}"
    elif dataset_name == "shanghai":
        sample_path = f"./samples/{run_name}_Shanghai_T{args.total_interp_steps}"
        metrics_path = f"./assets/eval_shanghai_{run_name}.txt"
        header = f"{run_name}_Shanghai"
    else:
        sample_path = f"./samples/{run_name}_SST_T{args.total_interp_steps}"
        metrics_path = f"./assets/eval_sst_{run_name}.txt"
        header = f"{run_name}_SST"
    return sample_path, metrics_path, header


if __name__ == "__main__":
    args = get_eval_args_with_yaml()
    args.data_name = slomo_model.normalize_dataset_name(args.data_name)
    slomo_model.configure_time_grid(args.data_name, args.total_interp_steps)

    print("Loading the trained model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    flowComp = slomo_model.UNet(2, 4)
    ArbTimeFlowIntrp = slomo_model.UNet(12, 5)
    trainFlowBackWarp = slomo_model.backWarp(args.patch_size, args.patch_size, device)

    save_path, run_name = resolve_eval_checkpoint_path(args)
    print("Loading model from:", save_path)
    checkpoint = torch.load(save_path, weights_only=True)
    flowComp.load_state_dict(checkpoint["flowComp"])
    ArbTimeFlowIntrp.load_state_dict(checkpoint["ArbTimeFlowIntrp"])

    seed = 0
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)

    flowComp = flowComp.to(device)
    ArbTimeFlowIntrp = ArbTimeFlowIntrp.to(device)
    trainFlowBackWarp = trainFlowBackWarp.to(device)
    flowComp.eval()
    ArbTimeFlowIntrp.eval()

    print("Loading the test dataset...")
    test_set, dataset_name = build_eval_dataset(args)
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
    inference_time = []
    print(f"Number of batches: {len(testloader)}")

    sample_path, metrics_save_path, header = metrics_output_paths(args, run_name, dataset_name)
    os.makedirs("./samples", exist_ok=True)

    print("Starting evaluation...")
    with torch.no_grad():
        for i, (inputs, targets, cond_params) in enumerate(testloader):
            print(i)
            I0, I1 = inputs
            I0 = I0.to(device)
            I1 = I1.to(device)
            norm_stats = None
            if dataset_name == "sea_temp" and len(cond_params) > 2:
                norm_stats = cond_params[2].cpu().detach().numpy()

            preds = []
            gpu_inference_time = 0.0
            for ii in range(len(targets)):
                target_interp_step = torch.tensor(ii + 1, dtype=torch.float32, device=device)
                target_interp_step = target_interp_step.repeat(I0.shape[0])
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                prediction = run_super_slomo_step(
                    flowComp,
                    ArbTimeFlowIntrp,
                    trainFlowBackWarp,
                    I0,
                    I1,
                    target_interp_step,
                    device,
                )
                end_event.record()
                end_event.synchronize()
                gpu_inference_time += start_event.elapsed_time(end_event) / 1000.0
                preds.append(prediction.cpu().detach().numpy())
            inference_time.append(gpu_inference_time / len(targets))

            for j in range(preds[0].shape[0]):
                rfne_at_time = []
                cc_at_time = []
                ssim_at_time = []
                for p in range(len(targets)):
                    target = targets[p].cpu().detach().numpy()
                    prediction = preds[p]
                    if dataset_name == "sea_temp" and norm_stats is not None:
                        metric_target = inverse_z_score_sst(target, norm_stats, p + 1)
                        metric_prediction = inverse_z_score_sst(
                            prediction, norm_stats, p + 1
                        )
                    else:
                        metric_target = target
                        metric_prediction = prediction

                    error = (
                        np.linalg.norm(
                            metric_prediction[j, 0, :, :] - metric_target[j, 0, :, :]
                        )
                        / np.linalg.norm(metric_target[j, 0, :, :])
                    )
                    rfne_at_time.append(error)

                    cc = scipy.stats.pearsonr(
                        metric_prediction[j, 0, :, :].flatten(),
                        metric_target[j, 0, :, :].flatten(),
                    )[0]
                    cc_at_time.append(cc)

                    ssim = cal_ssim_like_shanghai(
                        prediction[j, 0, :, :],
                        target[j, 0, :, :],
                        data_range=DATA_RANGE_BY_DATASET[dataset_name],
                    )
                    ssim_at_time.append(ssim)

                rfne_error.append(rfne_at_time)
                r2s.append(cc_at_time)
                ssim_lst.append(ssim_at_time)
            print(np.mean(np.vstack(r2s), axis=0))

            if i == 0:
                samples = {
                    "conditioning_snapshots": I0.cpu().detach().numpy(),
                    "targets": targets,
                    "predictions": preds,
                }
                if norm_stats is not None:
                    samples["norm_stats"] = norm_stats
                np.save(sample_path + ".npy", samples)
                print("Generated samples saved...")

    avg_RFNE = np.mean(np.vstack(rfne_error), axis=0)
    print(f"Average RFNE={repr(avg_RFNE)}")

    avg_R2 = np.mean(np.vstack(r2s), axis=0)
    print(f"Average Pearson correlation coefficients={repr(avg_R2)}")

    avg_ssim = np.mean(np.vstack(ssim_lst), axis=0)
    print(f"Average SSIM value={repr(avg_ssim)}")

    avg_infer_time = np.mean(inference_time, axis=0)
    print(f"Average Inference Time={repr(avg_infer_time)}")

    metrics = {
        "avg_rfne_steps": avg_RFNE,
        "avg_r2_steps": avg_R2,
        "avg_ssim_steps": avg_ssim,
        "avg_rfne_value": np.mean(avg_RFNE, axis=0),
        "avg_r2_value": np.mean(avg_R2, axis=0),
        "avg_ssim_value": np.mean(avg_ssim, axis=0),
        "avg_run_time": avg_infer_time,
    }
    save_metrics(metrics=metrics, save_path=metrics_save_path, header=header)
