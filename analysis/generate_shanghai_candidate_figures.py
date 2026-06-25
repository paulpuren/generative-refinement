import argparse
import copy
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import torch
from torch_ema import ExponentialMovingAverage

try:
    import cmweather  # noqa: F401
except ImportError:
    cmweather = None


CANDIDATE_ROOTS = [Path.cwd(), Path.cwd().parent]
REPO_ROOT = next((path.resolve() for path in CANDIDATE_ROOTS if (path / "src").exists()), None)
if REPO_ROOT is None:
    raise RuntimeError("Could not locate repo root containing ./src")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.data_shanghai import BOUNDS, COLOR_MAP, PIXEL_SCALE
from eval_super_slomo import ShanghaiEval, run_super_slomo_step
from src.diffusion_model import DiffusionModel
from src.flex import FLEX
from src.unet import UNet
from src.utilities import materialize_checkpoint_modules
import src.super_slomo as slomo_model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_CONFIG = {
    "dataset_path": REPO_ROOT / "/global/cfs/cdirs/m4633/puren/interp_dm/shanghai/shanghai.h5",
    "patch_size": 128,
    "total_interp_steps": 8,
    "flex_time_steps": 10,
    "flex_model_size": "small",
    "flex_mlp_ratio": 2,
    "use_scalar_film": True,
    "use_spatial_cond": True,
    "spatial_cond_scales": [32, 16],
    "spatial_cond_mode": "gated",
    "prediction_type": "v",
    "unet_base_width": 128,
    "flex_checkpoint": REPO_ROOT / "checkpoints/checkpoint_Model_FLEX3_small_mlp2_Data_shanghai_Optim_adam_cosine_lr0.0001_epoch300_stride32_T8_TfixedFalse_best.pt",
    "unet_checkpoint": REPO_ROOT / "checkpoints/checkpoint_Model_UNet_Data_shanghai_Optim_adam_lr0.0001_epoch400_stride32_T8_TfixedFalse.pt",
    "super_slomo_checkpoint": REPO_ROOT / "checkpoints/checkpoint_Model_SuperSloMo_Data_shanghai_Optim_adam_lr0.0001_epoch400_patchsize128_stride32_T8_TfixedTrue_best.pt",
    "cache_dir": REPO_ROOT / "analysis",
    "force_refresh": False,
    "display_mode": "pixel_scale",
    "selected_steps": [3, 6],
    "display_colormap": "shanghai_radar",
    "output_dir": REPO_ROOT / "analysis/shanghai_candidate_figures",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate candidate Shanghai comparison figures for multiple samples."
    )
    parser.add_argument(
        "--sample-indices",
        type=int,
        nargs="*",
        default=None,
        help="Explicit sample indices to render. Overrides --start-index and --num-samples.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Starting sample index when --sample-indices is not provided.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of consecutive samples to render when --sample-indices is not provided.",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Re-run inference even if cached .npz files already exist.",
    )
    parser.add_argument(
        "--display-mode",
        choices=["normalized", "raw_uint8", "pixel_scale"],
        default=BASE_CONFIG["display_mode"],
    )
    parser.add_argument(
        "--display-colormap",
        default=BASE_CONFIG["display_colormap"],
        help="Colormap name: shanghai_radar, ChaseSpectral, Carbone11, green_yellow, met_radar, turbo, viridis.",
    )
    parser.add_argument(
        "--selected-steps",
        type=int,
        nargs=2,
        default=BASE_CONFIG["selected_steps"],
        metavar=("STEP_A", "STEP_B"),
        help="Two interpolation step indices to plot for UNet and FLEX.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_CONFIG["output_dir"],
        help="Directory where candidate figures will be saved.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> dict:
    config = copy.deepcopy(BASE_CONFIG)
    config["force_refresh"] = args.force_refresh
    config["display_mode"] = args.display_mode
    config["display_colormap"] = args.display_colormap
    config["selected_steps"] = list(args.selected_steps)
    config["output_dir"] = args.output_dir.resolve()
    config["cache_dir"] = BASE_CONFIG["cache_dir"].resolve()
    config["dataset_path"] = Path("/global/cfs/cdirs/m4633/puren/interp_dm/shanghai/shanghai.h5")
    return config


def validate_paths(config: dict) -> None:
    for key in [
        "dataset_path",
        "flex_checkpoint",
        "unet_checkpoint",
        "super_slomo_checkpoint",
    ]:
        path = Path(config[key])
        if not path.exists():
            raise FileNotFoundError(f"Missing required path for {key}: {path}")
    config["cache_dir"].mkdir(parents=True, exist_ok=True)
    config["output_dir"].mkdir(parents=True, exist_ok=True)


def load_flex_model(config: dict, device: torch.device):
    encoder, task_encoder, task_encoder_end, decoder = FLEX(
        image_size=config["patch_size"],
        in_channels=1,
        out_channels=1,
        model_size=config["flex_model_size"],
        mlp_ratio=config["flex_mlp_ratio"],
        use_scalar_film=config["use_scalar_film"],
        use_spatial_cond=config["use_spatial_cond"],
        spatial_cond_scales=config["spatial_cond_scales"],
        spatial_cond_mode=config["spatial_cond_mode"],
    )
    model = DiffusionModel(
        encoder=encoder.to(device),
        decoder=decoder.to(device),
        task_encoder=task_encoder.to(device),
        task_encoder_end=task_encoder_end.to(device) if task_encoder_end is not None else None,
        diff_steps=config["flex_time_steps"],
        prediction_type=config["prediction_type"],
        criterion=torch.nn.L1Loss(),
        dt_normalization_scale=float(config["total_interp_steps"]),
    )
    checkpoint = torch.load(config["flex_checkpoint"], map_location="cpu", weights_only=True)
    materialize_checkpoint_modules(model, checkpoint["model"])
    model.load_state_dict(checkpoint["model"])
    ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
    if "ema" in checkpoint:
        ema.load_state_dict(checkpoint["ema"])
    model.eval()
    return model, ema


def load_unet_model(config: dict, device: torch.device):
    model = UNet(
        image_size=config["patch_size"],
        in_channels=2,
        out_channels=1,
        base_width=config["unet_base_width"],
    ).to(device)
    checkpoint = torch.load(config["unet_checkpoint"], map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def load_super_slomo_model(config: dict, device: torch.device):
    slomo_model.configure_time_grid("shanghai", config["total_interp_steps"])
    flow_comp = slomo_model.UNet(2, 4).to(device)
    flow_interp = slomo_model.UNet(12, 5).to(device)
    back_warp = slomo_model.backWarp(
        config["patch_size"], config["patch_size"], str(device)
    ).to(device)

    checkpoint = torch.load(
        config["super_slomo_checkpoint"], map_location="cpu", weights_only=True
    )
    flow_comp.load_state_dict(checkpoint["flowComp"])
    flow_interp.load_state_dict(checkpoint["ArbTimeFlowIntrp"])
    flow_comp.eval()
    flow_interp.eval()
    return flow_comp, flow_interp, back_warp


def tensor_to_image(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().float().squeeze().numpy()
    return np.asarray(x)


def predict_all_steps(
    config: dict,
    dataset: ShanghaiEval,
    sample_index: int,
    flex_model,
    flex_ema,
    unet_model,
    slomo_flow_comp,
    slomo_flow_interp,
    slomo_back_warp,
    device: torch.device,
) -> dict:
    inputs, targets, cond_params = dataset[sample_index]
    condition_start, condition_end = inputs

    condition_start = condition_start.unsqueeze(0).to(device)
    condition_end = condition_end.unsqueeze(0).to(device)
    total_interp_steps, reynolds_number = cond_params
    total_interp_steps = total_interp_steps.unsqueeze(0).to(device)
    reynolds_number = reynolds_number.unsqueeze(0).to(device)

    gt_frames = [tensor_to_image(target) for target in targets]
    flex_preds = []
    unet_preds = []
    slomo_preds = []

    with torch.no_grad():
        with flex_ema.average_parameters():
            for step in range(1, config["total_interp_steps"] + 1):
                target_interp_step = torch.tensor([step], dtype=torch.float32, device=device)

                flex_pred = flex_model.sample(
                    n_sample=1,
                    size=(1, config["patch_size"], config["patch_size"]),
                    cond_snapshot_start=condition_start,
                    cond_snapshot_end=condition_end,
                    fluid_condition=reynolds_number,
                    target_interp_step=target_interp_step,
                    total_interp_steps=total_interp_steps,
                    device=str(device),
                )
                flex_preds.append(tensor_to_image(flex_pred))

                unet_pred = unet_model.sample(
                    condition_start,
                    condition_end,
                    fluid_condition=reynolds_number,
                    target_interp_step=target_interp_step,
                    total_interp_steps=total_interp_steps,
                )
                unet_preds.append(tensor_to_image(unet_pred))

                slomo_pred = run_super_slomo_step(
                    slomo_flow_comp,
                    slomo_flow_interp,
                    slomo_back_warp,
                    condition_start,
                    condition_end,
                    target_interp_step,
                    str(device),
                )
                slomo_preds.append(tensor_to_image(slomo_pred))

    return {
        "sample_index": sample_index,
        "start": tensor_to_image(condition_start),
        "end": tensor_to_image(condition_end),
        "ground_truth": gt_frames,
        "flex": flex_preds,
        "unet": unet_preds,
        "super_slomo": slomo_preds,
    }


def save_comparison(result: dict, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        save_path,
        sample_index=np.array(result["sample_index"], dtype=np.int64),
        start=np.asarray(result["start"], dtype=np.float32),
        end=np.asarray(result["end"], dtype=np.float32),
        ground_truth=np.stack(result["ground_truth"], axis=0).astype(np.float32),
        flex=np.stack(result["flex"], axis=0).astype(np.float32),
        unet=np.stack(result["unet"], axis=0).astype(np.float32),
        super_slomo=np.stack(result["super_slomo"], axis=0).astype(np.float32),
    )


def load_comparison(save_path: Path) -> dict:
    loaded = np.load(save_path, allow_pickle=False)
    return {
        "sample_index": int(loaded["sample_index"]),
        "start": loaded["start"],
        "end": loaded["end"],
        "ground_truth": [frame for frame in loaded["ground_truth"]],
        "flex": [frame for frame in loaded["flex"]],
        "unet": [frame for frame in loaded["unet"]],
        "super_slomo": [frame for frame in loaded["super_slomo"]],
    }


def convert_frame_for_display(frame, display_mode: str):
    frame = np.asarray(frame, dtype=np.float32)
    if display_mode == "normalized":
        return np.clip(frame, 0.0, 1.0)
    if display_mode == "raw_uint8":
        return np.clip(frame * 255.0, 0.0, 255.0)
    if display_mode == "pixel_scale":
        return np.clip(frame * PIXEL_SCALE, 0.0, PIXEL_SCALE)
    raise ValueError(f"Unsupported display_mode: {display_mode}")


def get_display_range_and_label(display_mode: str):
    if display_mode == "normalized":
        return 0.0, 1.0, "Normalized intensity"
    if display_mode == "raw_uint8":
        return 0.0, 255.0, "Encoded radar value (0-255)"
    if display_mode == "pixel_scale":
        return 0.0, PIXEL_SCALE, f"Radar intensity (repo scale 0-{int(PIXEL_SCALE)})"
    raise ValueError(f"Unsupported display_mode: {display_mode}")


def get_shanghai_bounds_for_display(display_mode: str):
    bounds = np.asarray(BOUNDS, dtype=np.float32)
    if display_mode == "pixel_scale":
        return bounds
    if display_mode == "normalized":
        return bounds / PIXEL_SCALE
    if display_mode == "raw_uint8":
        return bounds * (255.0 / PIXEL_SCALE)
    raise ValueError(f"Unsupported display_mode: {display_mode}")


def validate_selected_steps(selected_steps, total_interp_steps: int):
    if len(selected_steps) != 2:
        raise ValueError("selected_steps must contain exactly two step indices.")
    for step in selected_steps:
        if step < 1 or step > total_interp_steps:
            raise ValueError(
                f"Selected step {step} is out of range 1..{total_interp_steps}."
            )
    return list(selected_steps)


def get_colormap_and_norm(config: dict):
    vmin, vmax, colorbar_label = get_display_range_and_label(config["display_mode"])

    if config["display_colormap"] == "ChaseSpectral":
        cmap = plt.get_cmap("ChaseSpectral")
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        return cmap, norm, colorbar_label

    if config["display_colormap"] == "Carbone11":
        cmap = plt.get_cmap("Carbone11")
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        return cmap, norm, colorbar_label

    if config["display_colormap"] == "green_yellow":
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "green_yellow",
            [
                "#081c15",
                "#1b4332",
                "#2d6a4f",
                "#52b788",
                "#95d5b2",
                "#d9ed92",
                "#f1fa8c",
                "#ffe66d",
            ],
            N=256,
        )
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        return cmap, norm, colorbar_label

    if config["display_colormap"] == "met_radar":
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "met_radar",
            [
                "#0b1f3a",
                "#145da0",
                "#00bcd4",
                "#39d353",
                "#d4e157",
                "#ffd54f",
                "#ff8f00",
                "#e53935",
                "#8e24aa",
            ],
            N=256,
        )
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        return cmap, norm, colorbar_label

    if config["display_colormap"] == "shanghai_radar":
        cmap = mcolors.ListedColormap(COLOR_MAP)
        bounds = get_shanghai_bounds_for_display(config["display_mode"])
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        return cmap, norm, colorbar_label

    if config["display_colormap"] == "turbo":
        cmap = plt.get_cmap("turbo")
    elif config["display_colormap"] == "viridis":
        cmap = plt.get_cmap("viridis")
    else:
        raise ValueError(f"Unsupported display_colormap: {config['display_colormap']}")

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    return cmap, norm, colorbar_label


def visualize_comparison(result: dict, config: dict, save_path: Path) -> None:
    selected_steps = validate_selected_steps(
        config["selected_steps"], len(result["ground_truth"])
    )

    display_mode = config["display_mode"]
    start_frame = convert_frame_for_display(result["start"], display_mode)
    end_frame = convert_frame_for_display(result["end"], display_mode)
    gt_frames = [
        convert_frame_for_display(frame, display_mode) for frame in result["ground_truth"]
    ]
    flex_frames = [
        convert_frame_for_display(frame, display_mode) for frame in result["flex"]
    ]
    unet_frames = [
        convert_frame_for_display(frame, display_mode) for frame in result["unet"]
    ]
    panels = [
        start_frame,
        end_frame,
        unet_frames[selected_steps[0] - 1],
        unet_frames[selected_steps[1] - 1],
        flex_frames[selected_steps[0] - 1],
        flex_frames[selected_steps[1] - 1],
    ]

    fig, axes = plt.subplots(
        1,
        6,
        figsize=(9.2, 1.95),
        facecolor="white",
    )

    cmap, norm, _ = get_colormap_and_norm(config)
    image_artist = None

    for ax, frame in zip(np.ravel(axes), panels):
        image_artist = ax.imshow(frame, cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.subplots_adjust(left=0.02, right=0.93, top=0.99, bottom=0.01, wspace=0.02)
    cbar = fig.colorbar(
        image_artist,
        ax=axes,
        orientation="vertical",
        fraction=0.03,
        pad=0.015,
        aspect=28,
    )
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=7.0, length=2.0, color="#444")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def resolve_sample_indices(args: argparse.Namespace) -> list[int]:
    if args.sample_indices:
        return args.sample_indices
    return list(range(args.start_index, args.start_index + args.num_samples))


def main() -> None:
    args = parse_args()
    config = build_config(args)
    validate_paths(config)

    sample_indices = resolve_sample_indices(args)
    print(f"Using device: {DEVICE}")
    print(f"Saving figures to: {config['output_dir']}")
    print(f"Sample indices: {sample_indices}")

    dataset = ShanghaiEval(
        total_interp_steps=config["total_interp_steps"],
        data_path=str(config["dataset_path"]),
        img_size=config["patch_size"],
        split="test",
    )
    print(f"Shanghai test samples: {len(dataset)}")

    flex_model, flex_ema = load_flex_model(config, DEVICE)
    unet_model = load_unet_model(config, DEVICE)
    slomo_flow_comp, slomo_flow_interp, slomo_back_warp = load_super_slomo_model(
        config, DEVICE
    )
    print("Loaded pretrained models.")

    for sample_index in sample_indices:
        per_sample_config = copy.deepcopy(config)
        cache_path = config["cache_dir"] / f"shanghai_sample_{sample_index}_comparison.npz"
        figure_path = config["output_dir"] / f"shanghai_sample_{sample_index:04d}_comparison.png"

        if cache_path.exists() and not config["force_refresh"]:
            comparison = load_comparison(cache_path)
            print(f"[sample {sample_index}] Loaded cache: {cache_path.name}")
        else:
            comparison = predict_all_steps(
                per_sample_config,
                dataset,
                sample_index,
                flex_model,
                flex_ema,
                unet_model,
                slomo_flow_comp,
                slomo_flow_interp,
                slomo_back_warp,
                DEVICE,
            )
            save_comparison(comparison, cache_path)
            print(f"[sample {sample_index}] Saved cache: {cache_path.name}")

        visualize_comparison(comparison, per_sample_config, figure_path)
        print(f"[sample {sample_index}] Saved figure: {figure_path}")

    print("Done.")


if __name__ == "__main__":
    main()
