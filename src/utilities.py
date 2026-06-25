import os, sys
import re
import cv2
import torch
from torch_ema import ExponentialMovingAverage
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from src.lion import Lion
from src.unet import UNet
from src.flex import FLEX
from src.diffusion_model import DiffusionModel
from src.ablation.diffusion_model_residual import DiffusionModel as DiffusionModelResidual
from datasets.data_nskt_updated import NSKT, NSKT_eval
from datasets.data_shanghai import Shanghai
from datasets.data_sea_temp import InputHandle, InputHandleEval
from pathlib import Path
from datetime import datetime


def cal_ssim_like_shanghai(pred, true, data_range=255.0):
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    img1 = pred.astype(float)
    img2 = true.astype(float)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = kernel @ kernel.T

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean()

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

def load_checkpoint(
        save_path, 
        model, 
        optimizer, 
        device,
        load_optimizer_state: bool = True,
    ):
    if not os.path.exists(save_path):
        print(f"Unable to load from {save_path}")

    checkpoint = torch.load(save_path, weights_only=True)
    materialize_checkpoint_modules(model, checkpoint["model"])
    model.load_state_dict(checkpoint["model"])
    ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
    if "ema" in checkpoint:
        try:
            ema.load_state_dict(checkpoint["ema"])
        except ValueError as exc:
            print(
                "EMA state incompatible with current model parameters; "
                f"reinitializing EMA. Details: {exc}"
            )
    if load_optimizer_state and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    elif not load_optimizer_state:
        print("Skipping optimizer state load; using a fresh optimizer for finetuning.")

    print(f"Loaded model from {save_path}")
    return model, ema, optimizer, int(checkpoint.get("epoch", 0))
          

def load_train_objs(args):
    # load training set
    if args.data_name == "nskt":
        train_set = NSKT(
            patch_size = args.patch_size, 
            crop_size = args.crop_size,
            stride = args.stride,
            num_interp_steps= args.total_interp_steps_train,
            scratch_dir = args.scratch_dir,
            flag = "train",
            is_T_fixed = args.is_T_fixed
        )
        val_set = NSKT(
            patch_size = args.patch_size, 
            crop_size = args.crop_size,
            stride = args.stride,
            num_interp_steps= args.total_interp_steps_train,
            scratch_dir = args.scratch_dir,
            flag = "valid",
            is_T_fixed = args.is_T_fixed
        )
    #
    elif args.data_name == "shanghai":
        # ['train', 'test', 'val']
        train_set = Shanghai(
            data_path = args.scratch_dir,
            img_size = args.patch_size, 
            type = "train",
            trans = None,
            total_interp_steps = args.total_interp_steps_train
        )
        val_set = Shanghai(
            data_path = args.scratch_dir,
            img_size = args.patch_size, 
            type = "val",
            trans = None,
            total_interp_steps = args.total_interp_steps_train
        )
    #
    elif args.data_name == "sea_temp":
        train_input_param = {
            'path': args.scratch_dir,
            'total_length': args.total_interp_steps_train, # total length of each sample (input + output)
            'input_length': 2, # length of input sequence
            'type': 'train', # train/test/valid
            'input_data_type': 'float32'
        }
        val_input_param = {
            'path': args.scratch_dir,
            'total_length': args.total_interp_steps_train, # total length of each sample (input + output)
            'input_length': 2, # length of input sequence
            'type': 'valid', # train/test/valid
            'input_data_type': 'float32'
        }
        train_set = InputHandle(train_input_param)
        val_set = InputHandle(val_input_param)
    # 
    else:
        print(
            "This dataset is not supported. We currently only support (nskt), (shanghai), and (sea_temp) datasets."
        )
        sys.exit()

    ema = None # placeholder for non-diffusion-based models
    if args.model in {'FLEX', 'FLEXResidual', 'FLEXResidualZero'}:
        dt_normalization_scale = float(args.total_interp_steps_train)
        encoder, task_encoder, task_encoder_end, decoder = FLEX(
            image_size = args.patch_size,
            in_channels = 1,
            out_channels = 1,
            model_size = args.flex_model_size,
            mlp_ratio = args.flex_mlp_ratio,
            use_scalar_film = args.use_scalar_film,
            use_spatial_cond = args.use_spatial_cond,
            spatial_cond_scales = args.spatial_cond_scales,
            spatial_cond_mode = args.spatial_cond_mode,
        )
        diffusion_cls = (
            DiffusionModelResidual
            if args.model in {'FLEXResidual', 'FLEXResidualZero'}
            else DiffusionModel
        )
        model = diffusion_cls(
            encoder = encoder.cuda(),
            decoder = decoder.cuda(),
            task_encoder = task_encoder.cuda(),
            task_encoder_end = task_encoder_end.cuda() if task_encoder_end is not None else None,
            diff_steps = args.time_steps,
            prediction_type = args.prediction_type,
            criterion = torch.nn.L1Loss(),
            dt_normalization_scale = dt_normalization_scale,
            condition_on_re = args.condition_on_re,
            condition_on_total_interp_steps = args.condition_on_total_interp_steps,
        )
        ema = ExponentialMovingAverage(
            model.parameters(),
            decay = 0.999
        )
    elif args.model == 'CrossFLEX':
        from src.crossflex import CrossFLEX
        from src.diffusion_model_crossflex import DiffusionModel as DiffusionModelCrossFLEX

        encoder, task_encoder, task_encoder_end, decoder = CrossFLEX(
            image_size = args.patch_size,
            in_channels = 1,
            out_channels = 1,
            model_size = args.flex_model_size,
        )
        model = DiffusionModelCrossFLEX(
            encoder = encoder.cuda(),
            decoder = decoder.cuda(),
            task_encoder = task_encoder.cuda(),
            task_encoder_end = task_encoder_end.cuda(),
            diff_steps = args.time_steps,
            prediction_type = args.prediction_type,
            criterion = torch.nn.L1Loss()
        )
        # choose optimizer
        ema = ExponentialMovingAverage(
            model.parameters(), 
            decay = 0.999
        )
    elif args.model == 'UNet':
        model = UNet(
            image_size = args.patch_size, 
            in_channels = 2, # start and end frames
            out_channels = 1, # predict interpolated frame
            base_width = args.base_width
        )
    else:
        print("This model is not supported.")
        sys.exit()

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr = args.learning_rate
        )
    elif args.optimizer == 'lion':
        optimizer = Lion(
            model.parameters(), 
            lr = args.learning_rate
        )
    else:
        print("Only Adam and Lion are supported.")
        sys.exit()
    return train_set, val_set, model, optimizer, ema


def load_eval_obj(args):
    # load evaluation set
    if args.data_name == "nskt":
        eval_set = NSKT_eval(
            patch_size=args.patch_size,
            crop_size=args.crop_size,
            stride=args.stride,
            num_interp_steps=args.total_interp_steps,
            re_id=args.re_id,
            scratch_dir=args.scratch_dir,
        )
    elif args.data_name == "shanghai":
        eval_set = Shanghai(
            data_path=args.scratch_dir,
            img_size=args.patch_size,
            type="test",
            trans=None,
            total_interp_steps=args.total_interp_steps,
        )
    elif args.data_name == "sea_temp":
        eval_input_param = {
            "path": args.scratch_dir,
            "total_length": args.total_interp_steps,
            "input_length": 2,
            "type": "test",
            "input_data_type": "float32",
        }
        eval_set = InputHandleEval(eval_input_param)
    else:
        print(
            "This dataset is not supported. We currently only support (nskt), (shanghai), and (sea_temp/sst) datasets."
        )
        sys.exit()

    ema = None  # placeholder for non-diffusion-based models
    if args.model in {"FLEX", "FLEXResidual", "FLEXResidualZero"}:
        dt_normalization_scale = float(args.total_interp_steps)
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
        diffusion_cls = (
            DiffusionModelResidual
            if args.model in {"FLEXResidual", "FLEXResidualZero"}
            else DiffusionModel
        )
        model = diffusion_cls(
            encoder=encoder.cuda(),
            decoder=decoder.cuda(),
            task_encoder=task_encoder.cuda(),
            task_encoder_end=task_encoder_end.cuda() if task_encoder_end is not None else None,
            diff_steps=args.time_steps,
            prediction_type=args.prediction_type,
            criterion=torch.nn.L1Loss(),
            dt_normalization_scale=dt_normalization_scale,
            condition_on_re=args.condition_on_re,
            condition_on_total_interp_steps=args.condition_on_total_interp_steps,
        )
        ema = ExponentialMovingAverage(
            model.parameters(),
            decay=0.999,
        )
    elif args.model == "CrossFLEX":
        from src.crossflex import CrossFLEX
        from src.diffusion_model_crossflex import DiffusionModel as DiffusionModelCrossFLEX

        encoder, task_encoder, task_encoder_end, decoder = CrossFLEX(
            image_size=args.patch_size,
            in_channels=1,
            out_channels=1,
            model_size=args.flex_model_size,
        )
        model = DiffusionModelCrossFLEX(
            encoder=encoder.cuda(),
            decoder=decoder.cuda(),
            task_encoder=task_encoder.cuda(),
            task_encoder_end=task_encoder_end.cuda(),
            diff_steps=args.time_steps,
            prediction_type=args.prediction_type,
            criterion=torch.nn.L1Loss(),
        )
        ema = ExponentialMovingAverage(
            model.parameters(),
            decay=0.999,
        )
    elif args.model == "UNet":
        model = UNet(
            image_size=args.patch_size,
            in_channels=2,
            out_channels=1,
            base_width=args.base_width,
        )
    else:
        print("This model is not supported.")
        sys.exit()

    return eval_set, model, ema

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
    if hasattr(args, "condition_on_re") and not args.condition_on_re:
        condition_tag += "_noRe"
    if (
        hasattr(args, "condition_on_total_interp_steps")
        and not args.condition_on_total_interp_steps
    ):
        condition_tag += "_noT"

    if args.model in {"FLEX", "FLEXResidual", "FLEXResidualZero", "CrossFLEX"}:
        model_name = f"{args.model}{condition_tag}"
        if args.checkpoint_path == '':
            run_name = "Model_{}_{}_mlp{}_Data_{}_Optim_{}_cosine_lr{}_epoch{}_patchsize{}_stride{}_T{}_Tfixed{}".format(
                    model_name,
                    args.flex_model_size,
                    args.flex_mlp_ratio,
                    args.data_name,
                    args.optimizer,
                    args.learning_rate,
                    args.epochs,
                    args.patch_size,
                    args.stride,
                    args.total_interp_steps_train,
                    args.is_T_fixed
            )
        else:
            run_name = "Model_ft_{}_{}_mlp{}_Data_{}_Optim_{}_cosine_lr{}_epoch{}_patchsize{}_stride{}_T{}_Tfixed{}".format(
                    model_name,
                    args.flex_model_size,
                    args.flex_mlp_ratio,
                    args.data_name,
                    args.optimizer,
                    args.learning_rate,
                    args.epochs,
                    args.patch_size,
                    args.stride,
                    args.total_interp_steps_train,
                    args.is_T_fixed
            )
    else:
        model_name = f"{args.model}{condition_tag}"
        if args.checkpoint_path == '':
            run_name = "Model_{}_Data_{}_Optim_{}_lr{}_epoch{}_patchsize{}_stride{}_T{}_Tfixed{}".format(
                    model_name,
                    args.data_name,
                    args.optimizer,
                    args.learning_rate,
                    args.epochs,
                    args.patch_size,
                    args.stride,
                    args.total_interp_steps_train,
                    args.is_T_fixed
            ) 
        else:
            run_name = "Model_ft_{}_Data_{}_Optim_{}_lr{}_epoch{}_patchsize{}_stride{}_T{}_Tfixed{}".format(
                    model_name,
                    args.data_name,
                    args.optimizer,
                    args.learning_rate,
                    args.epochs,
                    args.patch_size,
                    args.stride,
                    args.total_interp_steps_train,
                    args.is_T_fixed
            )
    return run_name


def resolve_eval_checkpoint_path(args):
    run_name = get_run_name(args)
    checkpoint_dir = Path("./checkpoints")

    run_names = [run_name]
    legacy_run_name = re.sub(r"_patchsize\d+(?=_stride)", "", run_name)
    if legacy_run_name != run_name:
        run_names.append(legacy_run_name)

    candidates = []
    if getattr(args, "checkpoint_path", ""):
        candidates.append(Path(args.checkpoint_path))

    for candidate_run_name in run_names:
        candidates.extend(
            [
                checkpoint_dir / f"checkpoint_{candidate_run_name}_best.pt",
                checkpoint_dir / f"checkpoint_{candidate_run_name}_last.pt",
                checkpoint_dir / f"checkpoint_{candidate_run_name}.pt",
            ]
        )

    seen = set()
    deduped_candidates = []
    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate_str not in seen:
            seen.add(candidate_str)
            deduped_candidates.append(candidate)

    for candidate in deduped_candidates:
        if candidate.exists():
            return str(candidate), run_name

    if not getattr(args, "checkpoint_path", ""):
        fallback_candidates = []
        for candidate_run_name in run_names:
            epoch_agnostic_pattern = re.sub(
                r"_epoch\d+", "_epoch*", candidate_run_name, count=1
            )
            fallback_candidates.extend(
                checkpoint_dir.glob(f"checkpoint_{epoch_agnostic_pattern}_*.pt")
            )
        fallback_candidates = sorted(
            set(fallback_candidates),
            key=lambda path: (
                0 if path.name.endswith("_best.pt") else 1,
                0 if path.name.endswith("_last.pt") else 1,
                -path.stat().st_mtime,
            ),
        )
        if fallback_candidates:
            return str(fallback_candidates[0]), run_name

    raise FileNotFoundError(
        "No evaluation checkpoint found. Checked: "
        + ", ".join(str(candidate) for candidate in deduped_candidates)
    )

def save_metrics(
        metrics: dict, 
        save_path: str, 
        header: str = ""
    ):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with save_path.open("a", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if header:
            f.write(f"{header}\n")
        for k, v in metrics.items():
            if isinstance(v, float):
                f.write(f"{k}: {v:.6f}\n")
            else:
                f.write(f"{k}: {v}\n")
