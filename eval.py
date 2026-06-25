"""
Evaluation for NSKT fluid data
------------------------------
* eval for different Reynolds numbers
* eval for different interpolation steps
* eval for different patch sizes
* eval for different models
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from src.helper import *
from torch.utils.data import Dataset, DataLoader, Subset
from torch_ema import ExponentialMovingAverage
import scipy.stats
from datasets.data_nskt_updated import NSKT_eval
from utils.params_eval import get_args
from src.utilities import *

RE_EVAL_LIST = [
    600, 1000, 2000, 4000, 8000, 
    12000, 16000, 24000, 32000, 36000
]
NSKT_SSIM_DATA_RANGE = 1.0


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

if __name__ == "__main__":
    args = get_eval_args_with_yaml()
    
    # load model
    print("Loading the trained model...")

    _, model, ema = load_eval_obj(args=args)

    # model save path
    save_path, run_name = resolve_eval_checkpoint_path(args)
    print("Loading model from: ", save_path)
    checkpoint = torch.load(save_path, weights_only = True)
    model.load_state_dict(checkpoint["model"])
    if ema is not None:
        ema.load_state_dict(checkpoint["ema"])

    # set seed
    seed = 0
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)

    model.to('cuda')
    model.eval()

    # load test data
    print("Loading the test dataset...")
    test_set = NSKT_eval(
        patch_size = args.patch_size,
        crop_size = args.crop_size,
        stride = args.stride,
        num_interp_steps = args.total_interp_steps,
        re_id = args.re_id,
        scratch_dir = args.scratch_dir
    )
    full_eval_size = len(test_set)
    max_eval_samples = int(os.getenv("NSKT_EVAL_MAX_SAMPLES", "256"))
    if max_eval_samples > 0 and len(test_set) > max_eval_samples:
        rng = np.random.default_rng(seed)
        subset_indices = rng.choice(
            len(test_set), size=max_eval_samples, replace=False
        )
        subset_indices = np.sort(subset_indices).tolist()
        test_set = Subset(test_set, subset_indices)
        print(f"Using NSKT eval subset: {len(subset_indices)}/{full_eval_size} samples")
    testloader = DataLoader(
        test_set,
        batch_size = args.batch_size,
        pin_memory = True,
        shuffle = False,
        # sampler=DistributedSampler(dataset),
        num_workers = 8
    )

    RFNE_error = []
    residual_RFNE_error = []
    start_baseline_RFNE_error = []
    end_baseline_RFNE_error = []
    R2s = []
    ssim_lst = []
    print(f'Number of batches: {len(testloader)}')

    if ema is not None:
        with ema.average_parameters():
            with torch.no_grad():
                model.eval()
                inference_time = []
                for i, (inputs, targets, cond_params) in enumerate(testloader):
                    print(i)
                    # Unpack the input tuple
                    # [b,c,h,w] = [32, 1, 128, 128]
                    condition_start, condition_end = inputs
                    condition_start = condition_start.to('cuda')
                    condition_end = condition_end.to('cuda')
                    # print("condition_end shape: ", condition_end.shape) 
                    # print("lengths of targets: ", len(targets)) # 20
                    # print("target shape: ", targets[0].shape) # [32,1,128,128]
                    
                    # unpack the condition parameters
                    total_interp_steps, reynolds_number = cond_params
                    reynolds_number = reynolds_number.to('cuda') 
                    total_interp_steps = total_interp_steps.to('cuda')

                    preds = []
                    len_targets = len(targets)

                    gpu_inference_time = 0.0
                    for ii in range(len(targets)): # total interp step (e.g., 20)
                        target_interp_step = torch.tensor(
                            (ii + 1), 
                            dtype = torch.float32
                        ).to('cuda')
                        target_interp_step = target_interp_step.repeat(
                            condition_start.shape[0]
                        )
                        start_event = torch.cuda.Event(enable_timing=True)
                        end_event = torch.cuda.Event(enable_timing=True)
                        start_event.record()
                        predictions = model.sample(
                            condition_start.shape[0],
                            (1, args.patch_size, args.patch_size),
                            condition_start, 
                            condition_end, 
                            reynolds_number,
                            target_interp_step,
                            total_interp_steps,
                            'cuda'
                        ) # shape: [b, c, h, w] = [32, 1, 128, 128]    
                        end_event.record()
                        end_event.synchronize()
                        gpu_inference_time += start_event.elapsed_time(end_event) / 1000.0
                        # print("predictions shape: ", predictions.shape)             
                        preds.append(predictions.cpu().detach().numpy())
                    inference_time.append(gpu_inference_time / len(targets)) # each snapshot

                    # iterate over batch size: 32
                    condition_start_np = condition_start.cpu().detach().numpy()
                    condition_end_np = condition_end.cpu().detach().numpy()
                    for j in range(predictions.shape[0]):
                        RFNE_error_at_time_p = []
                        residual_RFNE_error_at_time_p = []
                        start_baseline_RFNE_error_at_time_p = []
                        end_baseline_RFNE_error_at_time_p = []
                        cc_error_at_time_p = []
                        ssim_error_at_time_p = []
                        
                        # total interp steps
                        for p in range(len(targets)): 

                            # data shape: [b,c,h,w]
                            target = targets[p].cpu().detach().numpy()
                            prediction = preds[p]

                            # compute RFNE
                            error = (
                                np.linalg.norm(
                                    prediction[j, 0, :, :] - target[j, 0, :, :]
                                ) / \
                                np.linalg.norm(
                                    target[j, 0, :, :]
                                )
                            )
                            RFNE_error_at_time_p.append(error)
                            residual_error = (
                                np.linalg.norm(
                                    prediction[j, 0, :, :] - target[j, 0, :, :]
                                ) /
                                max(
                                    np.linalg.norm(
                                        target[j, 0, :, :] - condition_start_np[j, 0, :, :]
                                    ),
                                    1e-12,
                                )
                            )
                            residual_RFNE_error_at_time_p.append(residual_error)
                            start_baseline_error = (
                                np.linalg.norm(
                                    condition_start_np[j, 0, :, :] - target[j, 0, :, :]
                                ) /
                                np.linalg.norm(target[j, 0, :, :])
                            )
                            start_baseline_RFNE_error_at_time_p.append(start_baseline_error)
                            end_baseline_error = (
                                np.linalg.norm(
                                    condition_end_np[j, 0, :, :] - target[j, 0, :, :]
                                ) /
                                np.linalg.norm(target[j, 0, :, :])
                            )
                            end_baseline_RFNE_error_at_time_p.append(end_baseline_error)

                            # compute correlation coefficient
                            cc = scipy.stats.pearsonr(
                                prediction[j, 0, :, :].flatten(), 
                                target[j, 0, :, :].flatten()
                            )[0]
                            cc_error_at_time_p.append(cc)

                            ssim = cal_ssim_like_shanghai(
                                prediction[j, 0, :, :],
                                target[j, 0, :, :],
                                data_range=NSKT_SSIM_DATA_RANGE,
                            )
                            ssim_error_at_time_p.append(ssim)

                        RFNE_error.append(RFNE_error_at_time_p)
                        residual_RFNE_error.append(residual_RFNE_error_at_time_p)
                        start_baseline_RFNE_error.append(start_baseline_RFNE_error_at_time_p)
                        end_baseline_RFNE_error.append(end_baseline_RFNE_error_at_time_p)
                        R2s.append(cc_error_at_time_p)
                        ssim_lst.append(ssim_error_at_time_p)
                    print(np.mean(np.vstack(R2s), axis=0 ))

                    if i == 0:
                        samples = {
                            'conditioning_snapshots': condition_start.cpu().detach().numpy(),
                            'targets': targets,
                            'predictions': preds
                        }

                        if not os.path.exists("./samples"):
                            os.makedirs("./samples")
                        sample_path = "./samples/{}_RE{}_T{}".format(
                            run_name,
                            RE_EVAL_LIST[args.re_id],
                            args.total_interp_steps
                        )
                        np.save(sample_path + '.npy', samples)
                        print('Generated samples saved...')
    else:
        with torch.no_grad():
            model.eval()
            inference_time = []
            for i, (inputs, targets, cond_params) in enumerate(testloader):
                print(i)
                # Unpack the input tuple
                # [b,c,h,w] = [32, 1, 128, 128]
                condition_start, condition_end = inputs
                condition_start = condition_start.to('cuda')
                condition_end = condition_end.to('cuda')
    
                # unpack the condition parameters
                total_interp_steps, reynolds_number = cond_params
                reynolds_number = reynolds_number.to('cuda') 
                total_interp_steps = total_interp_steps.to('cuda')

                preds = []
                len_targets = len(targets)

                gpu_inference_time = 0.0
                for ii in range(len(targets)): # total interp step (e.g., 20)
                    target_interp_step = torch.tensor(
                        (ii + 1), 
                        dtype = torch.float32
                    ).to('cuda')
                    target_interp_step = target_interp_step.repeat(
                        condition_start.shape[0]
                    )
                    # shape: [b, c, h, w] = [32, 1, 128, 128]    
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    predictions = model.sample(
                        condition_start,
                        condition_end,
                        reynolds_number,
                        target_interp_step,
                        total_interp_steps
                    )
                    end_event.record()
                    end_event.synchronize()
                    gpu_inference_time += start_event.elapsed_time(end_event) / 1000.0
                    preds.append(predictions.cpu().detach().numpy())
                inference_time.append(gpu_inference_time / len(targets)) # each snapshot

                # iterate over batch size: 32
                condition_start_np = condition_start.cpu().detach().numpy()
                condition_end_np = condition_end.cpu().detach().numpy()
                for j in range(predictions.shape[0]):
                    RFNE_error_at_time_p = []
                    residual_RFNE_error_at_time_p = []
                    start_baseline_RFNE_error_at_time_p = []
                    end_baseline_RFNE_error_at_time_p = []
                    cc_error_at_time_p = []
                    ssim_error_at_time_p = []
                    
                    # total interp steps
                    for p in range(len(targets)): 

                        # data shape: [b,c,h,w]
                        target = targets[p].cpu().detach().numpy()
                        prediction = preds[p]

                        # compute RFNE
                        error = (
                            np.linalg.norm(
                                prediction[j, 0, :, :] - target[j, 0, :, :]
                            ) / \
                            np.linalg.norm(
                                target[j, 0, :, :]
                            )
                        )
                        RFNE_error_at_time_p.append(error)
                        residual_error = (
                            np.linalg.norm(
                                prediction[j, 0, :, :] - target[j, 0, :, :]
                            ) /
                            max(
                                np.linalg.norm(
                                    target[j, 0, :, :] - condition_start_np[j, 0, :, :]
                                ),
                                1e-12,
                            )
                        )
                        residual_RFNE_error_at_time_p.append(residual_error)
                        start_baseline_error = (
                            np.linalg.norm(
                                condition_start_np[j, 0, :, :] - target[j, 0, :, :]
                            ) /
                            np.linalg.norm(target[j, 0, :, :])
                        )
                        start_baseline_RFNE_error_at_time_p.append(start_baseline_error)
                        end_baseline_error = (
                            np.linalg.norm(
                                condition_end_np[j, 0, :, :] - target[j, 0, :, :]
                            ) /
                            np.linalg.norm(target[j, 0, :, :])
                        )
                        end_baseline_RFNE_error_at_time_p.append(end_baseline_error)

                        # compute correlation coefficient
                        cc = scipy.stats.pearsonr(
                            prediction[j, 0, :, :].flatten(), 
                            target[j, 0, :, :].flatten()
                        )[0]
                        cc_error_at_time_p.append(cc)

                        ssim = cal_ssim_like_shanghai(
                            prediction[j, 0, :, :],
                            target[j, 0, :, :],
                            data_range=NSKT_SSIM_DATA_RANGE,
                        )
                        ssim_error_at_time_p.append(ssim)

                    RFNE_error.append(RFNE_error_at_time_p)
                    residual_RFNE_error.append(residual_RFNE_error_at_time_p)
                    start_baseline_RFNE_error.append(start_baseline_RFNE_error_at_time_p)
                    end_baseline_RFNE_error.append(end_baseline_RFNE_error_at_time_p)
                    R2s.append(cc_error_at_time_p)
                    ssim_lst.append(ssim_error_at_time_p)
                print(np.mean(np.vstack(R2s), axis=0 ))

                if i == 0:
                    samples = {
                        'conditioning_snapshots': condition_start.cpu().detach().numpy(),
                        'targets': targets,
                        'predictions': preds
                    }

                    if not os.path.exists("./samples"):
                        os.makedirs("./samples")
                    sample_path = "./samples/{}_RE{}_T{}".format(
                        run_name,
                        RE_EVAL_LIST[args.re_id],
                        args.total_interp_steps
                    )
                    np.save(sample_path + '.npy', samples)
                    print('Generated samples saved...')

    avg_RFNE = np.mean(np.vstack(RFNE_error), axis=0)
    print(f'Average RFNE={repr(avg_RFNE)}')

    avg_residual_RFNE = np.mean(np.vstack(residual_RFNE_error), axis=0)
    print(f'Average residual-normalized RFNE={repr(avg_residual_RFNE)}')

    avg_start_baseline_RFNE = np.mean(np.vstack(start_baseline_RFNE_error), axis=0)
    print(f'Average start baseline RFNE={repr(avg_start_baseline_RFNE)}')

    avg_end_baseline_RFNE = np.mean(np.vstack(end_baseline_RFNE_error), axis=0)
    print(f'Average end baseline RFNE={repr(avg_end_baseline_RFNE)}')

    avg_R2 = np.mean(np.vstack(R2s), axis=0)
    print(f'Average Pearson correlation coefficients={repr(avg_R2)}')

    avg_ssim = np.mean(np.vstack(ssim_lst), axis=0)
    print(f"Average SSIM value={repr(avg_ssim)}")

    # print("inference time shape: ", len(inference_time))
    # print("inference time samples: ", inference_time[0].shape)
    avg_infer_time = np.mean(inference_time, axis=0)
    print(f'Average Inference Time={repr(avg_infer_time)}')

    # save results
    metrics = {
        "avg_rfne_steps": avg_RFNE, 
        "avg_residual_rfne_steps": avg_residual_RFNE,
        "avg_start_baseline_rfne_steps": avg_start_baseline_RFNE,
        "avg_end_baseline_rfne_steps": avg_end_baseline_RFNE,
        "avg_r2_steps": avg_R2,
        "avg_ssim_steps": avg_ssim,
        "avg_rfne_value": np.mean(avg_RFNE, axis=0), 
        "avg_residual_rfne_value": np.mean(avg_residual_RFNE, axis=0),
        "avg_start_baseline_rfne_value": np.mean(avg_start_baseline_RFNE, axis=0),
        "avg_end_baseline_rfne_value": np.mean(avg_end_baseline_RFNE, axis=0),
        "avg_r2_value": np.mean(avg_R2, axis=0),
        "avg_ssim_value": np.mean(avg_ssim, axis=0),
        "avg_run_time": avg_infer_time
    }
    metrics_save_path = "./assets/eval_re{}_{}.txt".format(
        RE_EVAL_LIST[args.re_id],
        run_name
    )
    header = "{}_RE{}".format(run_name, RE_EVAL_LIST[args.re_id])
    save_metrics(
        metrics = metrics, 
        save_path = metrics_save_path, 
        header = header # "model=resnet50, split=test"
    )



# export CUDA_VISIBLE_DEVICES=7; python evaluation.py --task forecast --batch-size 32 --horizen 50 --Reynolds-number 12000
