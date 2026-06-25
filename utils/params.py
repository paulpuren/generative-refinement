import argparse
def get_args():
    parser = argparse.ArgumentParser(
        description = "FLEX for Temporal Interpolation"
    )
    # general parameters
    parser.add_argument(
        "--run_name", 
        type = str, 
        default = 'run1', 
        help = "Name of the current run."
    )
    parser.add_argument(
        "--data_name", 
        type = str, 
        default = 'nskt', 
        help = "Name of the dataset."
    )
    parser.add_argument(
        "--sampling_freq", 
        default = 10, 
        type = int, 
        help = "How often to save a snapshot"
    )
    parser.add_argument(
        "--validation_freq",
        default = 1,
        type = int,
        help = "How often to run validation, in epochs"
    )
    # Training parameters
    parser.add_argument(
        "--optimizer", 
        type = str, 
        default = "adam", 
        help = "Optimizer: adam or lion"
    )
    parser.add_argument(
        "--epochs", 
        default = 200, 
        type = int, 
        help = "Total epochs to train the model"
    )
    parser.add_argument(
        "--batch_size", 
        default = 16, 
        type = int, 
        help = "Input batch size on each device (default: 32)"
    )
    parser.add_argument(
        "--learning_rate", 
        default = 2e-4, 
        type = float, 
        help = 'learning rate'
    ) # 1e-4 for adam; 1e-5 for lion
    parser.add_argument(
        "--num_workers",
        default = 16,
        type = int,
        help = "Number of DataLoader workers per rank"
    )
    parser.add_argument(
        "--prefetch_factor",
        default = 4,
        type = int,
        help = "Number of prefetched batches per worker when num_workers > 0"
    )
    parser.add_argument(
        "--persistent_workers",
        default = True,
        type = lambda x: (str(x).lower() == 'true'),
        help = "Keep DataLoader workers alive across epochs"
    )
    parser.add_argument(
        "--ddp_find_unused_parameters",
        default = False,
        type = lambda x: (str(x).lower() == 'true'),
        help = "Enable DDP unused-parameter detection for models with conditional branches"
    )
    parser.add_argument(
        "--checkpoint_path", 
        default = "", 
        type = str, 
        help = "for reloading checkpoint and keep training"
    )
    parser.add_argument(
        "--resume_training",
        default = False,
        type = lambda x: (str(x).lower() == 'true'),
        help = "Resume optimizer state and epoch count from checkpoint_path."
    )
    # dataset parameters
    parser.add_argument(
        "--total_interp_steps_train", 
        default=1, 
        type=int, 
        help='total interpolation steps to condition on'
    )
    parser.add_argument(
        "--is_T_fixed", 
        default = True,
        type = lambda x: (str(x).lower() == 'true'), 
        help = "fix or change T in training."
    )
    parser.add_argument(
        "--patch_size", 
        default = 256, 
        type = int, 
        help = "Patch size for the datasets"
    )
    parser.add_argument(
        "--crop_size",
        default = None,
        type = int,
        help = "Crop size for the raw dataset before optional downsampling to patch_size"
    )
    parser.add_argument(
        "--stride", 
        default = 128, 
        type = int, 
        help = "Stride for the datasets"
    )
    parser.add_argument(
        "--scratch_dir",
        default = "/global/cfs/cdirs/m4633/foundationmodel/nskt_tensor/", 
        type = str, 
        help = "Directory for the dataset"
    )
    # Diffusion parameters
    parser.add_argument(
        "--prediction_type", 
        type = str, 
        default = 'v', 
        help = "Quantity to predict during training."
    )
    parser.add_argument(
        "--sampler", 
        type = str, 
        default = 'ddim', 
        help = "Sampler to use to generate images"
    )    
    parser.add_argument(
        "--time_steps", 
        type = int, 
        default = 10, 
        help = "Diffusion time steps for sampling"
    )
    parser.add_argument(
        "--cond_drop_prob",
        type=int,
        default=0,
        help="Percent probability [0-100] of dropping conditioning during training.",
    )
    # model parameters
    parser.add_argument(
        "--model", 
        type = str, 
        default = 'FLEX', 
        help = "model: FLEX, FLEXResidual, FLEXResidualZero, CrossFLEX, UNet, or SuperSloMo"
    )    
    parser.add_argument(
        "--flex_model_size", 
        type = str, 
        default = 'small', 
        help = "model: small, medium, big"
    )  
    parser.add_argument(
        "--flex_mlp_ratio", 
        type = int, 
        default = 2, 
        help = "mlp ratios: 2 or 4"
    )   
    parser.add_argument(
        "--use_scalar_film",
        default = True,
        type = lambda x: (str(x).lower() == 'true'),
        help = "Apply scalar FiLM conditioning from tau, dt, and Re in FLEX."
    )
    parser.add_argument(
        "--use_spatial_cond",
        default = True,
        type = lambda x: (str(x).lower() == 'true'),
        help = "Inject shared spatial endpoint conditioning in FLEX."
    )
    parser.add_argument(
        "--spatial_cond_scales",
        nargs = "+",
        type = int,
        default = [32, 16],
        help = "Spatial resolutions where FLEX injects endpoint condition maps."
    )
    parser.add_argument(
        "--spatial_cond_mode",
        type = str,
        default = "gated",
        help = "Spatial conditioning mode for FLEX: gated or additive."
    )
    parser.add_argument(
        "--condition_on_re",
        default = True,
        type = lambda x: (str(x).lower() == 'true'),
        help = "Use Reynolds number conditioning. Set false for no-Re ablations."
    )
    parser.add_argument(
        "--condition_on_total_interp_steps",
        default = True,
        type = lambda x: (str(x).lower() == 'true'),
        help = "Use total interpolation steps conditioning. Set false for no-total-steps ablations."
    )
    # U-Net parameters
    parser.add_argument(
        "--base_width", 
        type = int, 
        default = 128, 
        help = "Basewidth of U-Net"
    )    
    return parser.parse_args()
