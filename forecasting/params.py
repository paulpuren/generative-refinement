import argparse


def str_to_bool(value):
    return str(value).lower() == "true"


def get_train_args():
    parser = argparse.ArgumentParser(description="FLEX forecasting diffusion training")
    parser.add_argument("--run_name", type=str, default="FLEXForecast")
    parser.add_argument("--data_name", type=str, default="forecasting_nskt")
    parser.add_argument("--sampling_freq", type=int, default=10)
    parser.add_argument("--validation_freq", type=int, default=5)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--learning_rate", type=float, default=1.0e-4)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--persistent_workers", type=str_to_bool, default=True)
    parser.add_argument("--ddp_find_unused_parameters", type=str_to_bool, default=True)
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--resume_training", type=str_to_bool, default=False)
    parser.add_argument("--total_interp_steps_train", type=int, default=1)
    parser.add_argument("--is_T_fixed", type=str_to_bool, default=True)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--crop_size", type=int, default=None)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument(
        "--scratch_dir",
        type=str,
        default="/global/cfs/cdirs/m4633/foundationmodel/nskt_tensor/",
    )
    parser.add_argument("--prediction_type", type=str, default="v")
    parser.add_argument("--sampler", type=str, default="ddim")
    parser.add_argument("--time_steps", type=int, default=3)
    parser.add_argument("--model", type=str, default="FLEXForecast")
    parser.add_argument("--flex_model_size", type=str, default="small")
    parser.add_argument("--flex_mlp_ratio", type=int, default=2)
    parser.add_argument("--use_scalar_film", type=str_to_bool, default=True)
    parser.add_argument("--use_spatial_cond", type=str_to_bool, default=True)
    parser.add_argument("--spatial_cond_scales", nargs="+", type=int, default=[32, 16])
    parser.add_argument("--spatial_cond_mode", type=str, default="gated")
    parser.add_argument("--condition_on_re", type=str_to_bool, default=True)
    parser.add_argument("--reynolds_normalization_scale", type=float, default=1.0)
    parser.add_argument("--condition_on_total_interp_steps", type=str_to_bool, default=True)
    parser.add_argument("--forecast_baseline", type=str, default="persistence")
    parser.add_argument("--forecast_train_horizon", type=int, default=1)
    parser.add_argument("--direct_horizon_loss_weight", type=float, default=1.0)
    parser.add_argument("--rollout_loss_weight", type=float, default=0.0)
    parser.add_argument("--rollout_batch_size", type=int, default=0)
    parser.add_argument("--forecast_conditioning_mode", type=str, default="forecast")
    parser.add_argument("--base_width", type=int, default=128)
    parser.add_argument("--wandb", type=str_to_bool, default=False)
    parser.add_argument("--wandb_project", type=str, default="InterpDM")
    return parser.parse_args()


def get_eval_args():
    parser = argparse.ArgumentParser(description="FLEX autoregressive forecasting evaluation")
    parser.add_argument("--model", type=str, default="FLEXForecast")
    parser.add_argument("--flex_model_size", type=str, default="small")
    parser.add_argument("--flex_mlp_ratio", type=int, default=2)
    parser.add_argument("--use_scalar_film", type=str_to_bool, default=True)
    parser.add_argument("--use_spatial_cond", type=str_to_bool, default=True)
    parser.add_argument("--spatial_cond_scales", nargs="+", type=int, default=[32, 16])
    parser.add_argument("--spatial_cond_mode", type=str, default="gated")
    parser.add_argument("--condition_on_re", type=str_to_bool, default=True)
    parser.add_argument("--reynolds_normalization_scale", type=float, default=1.0)
    parser.add_argument("--condition_on_total_interp_steps", type=str_to_bool, default=True)
    parser.add_argument("--forecast_baseline", type=str, default="persistence")
    parser.add_argument("--forecast_train_horizon", type=int, default=1)
    parser.add_argument("--forecast_conditioning_mode", type=str, default="forecast")
    parser.add_argument("--data_name", type=str, default="forecasting_nskt")
    parser.add_argument("--re_id", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--crop_size", type=int, default=None)
    parser.add_argument("--prediction_type", type=str, default="v")
    parser.add_argument("--sampler", type=str, default="ddim")
    parser.add_argument("--time_steps", type=int, default=3)
    parser.add_argument("--forecast_horizon", type=int, default=20)
    parser.add_argument("--total_interp_steps", type=int, default=20)
    parser.add_argument("--total_interp_steps_train", type=int, default=1)
    parser.add_argument("--base_width", type=int, default=128)
    parser.add_argument(
        "--scratch_dir",
        type=str,
        default="/global/cfs/cdirs/m4633/foundationmodel/nskt_tensor/",
    )
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning_rate", type=float, default=1.0e-4)
    parser.add_argument("--is_T_fixed", type=str_to_bool, default=True)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--checkpoint_path", type=str, default="")
    return parser.parse_args()
