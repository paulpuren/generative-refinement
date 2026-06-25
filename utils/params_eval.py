import argparse
from html import parser

def get_args():
    parser = argparse.ArgumentParser(
        description = 'Minimalistic Diffusion Model for Super-resolution'
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
    parser.add_argument(
        "--data_name", 
        type = str, 
        default = 'nskt', 
        help = "Name of the dataset."
    )
    # RE_EVAL_LIST = [
    #     600, 1000, 2000, 4000, 8000, 
    #     12000, 16000, 24000, 32000, 36000
    # ]
    parser.add_argument(
        '--re_id', 
        default = -1, 
        type = int,
        help = 'reynolds number id (0-7): check RE_EVAL_LIST'
    )
    parser.add_argument(
        '--batch_size', 
        default = 128, 
        type = int,
        help = 'Input batch size on each device (default: 32)'
    )
    parser.add_argument(
        '--patch_size', 
        default = 256, 
        type = int, 
        help = 'target resolution'
    )
    parser.add_argument(
        '--crop_size',
        default = None,
        type = int,
        help = 'crop size before optional downsampling to patch_size'
    )
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
        help = "Time steps for sampling"
    )
    parser.add_argument(
        "--cond_drop_prob",
        type=int,
        default=0,
        help="Percent probability [0-100] of dropping conditioning during training.",
    )
    parser.add_argument(
        '--total_interp_steps', 
        default = 1, 
        type = int,
        help = 'the interp steps used in evaluation'
    )
    parser.add_argument(
        '--total_interp_steps_train', 
        default = 1, 
        type = int,
        help = 'the interp steps used in training'
    )
    parser.add_argument(
        "--base_width", 
        type = int,
        default = 128, 
        help = "Basewidth of U-Net"
    )
    parser.add_argument(
        '--scratch_dir', 
        default = '/global/cfs/cdirs/m4633/foundationmodel/nskt_tensor/', 
        type = str, 
        help = 'Directory for the dataset'
    )
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
        "--learning_rate", 
        default = 2e-4, 
        type = float, 
        help = 'learning rate'
    )
    parser.add_argument(
        "--is_T_fixed", 
        default = True,
        type = lambda x: (str(x).lower() == 'true'), 
        help = "fix or change T in training."
    )
    parser.add_argument(
        "--stride", 
        default = 128, 
        type = int, 
        help = "Stride for the datasets"
    )
    parser.add_argument(
        "--checkpoint_path", 
        default = "", 
        type = str, 
        help = "for reloading checkpoint and keep training"
    )
    return parser.parse_args()
