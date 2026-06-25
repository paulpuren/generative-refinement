import argparse
import os, sys, time
import numpy as np
import wandb
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import yaml
from torch_ema import ExponentialMovingAverage
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, barrier, get_rank, is_initialized, all_reduce, get_world_size
from diffusers.optimization import get_linear_schedule_with_warmup as scheduler
from src.unet import UNet
# import src.super_slomo as model
import src.super_slomo as slomo_model

# from src.diffusion_model import DiffusionModel
from datasets.data_nskt import NSKT
from datasets.data_shanghai import Shanghai
from datasets.data_sea_temp import InputHandle
from src.lion import Lion
from src.plotting import plot_samples
from src.utilities import get_run_name

import warnings
warnings.filterwarnings("ignore")


def _get_args_from_argv(argv):
    saved_argv = sys.argv
    try:
        sys.argv = argv
        return build_parser().parse_args()
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


def get_args_with_yaml():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default="")
    pre_args, remaining = pre_parser.parse_known_args()

    prog = sys.argv[0]
    default_args = _get_args_from_argv([prog])
    cli_args = _get_args_from_argv([prog] + remaining)

    merged = argparse.Namespace(**vars(default_args))
    config_data = load_yaml_config(pre_args.config)

    valid_keys = set(vars(default_args).keys())
    invalid_keys = sorted(set(config_data.keys()) - valid_keys)
    if invalid_keys:
        raise KeyError(
            f"Unknown keys in config {pre_args.config}: {', '.join(invalid_keys)}"
        )

    for key, value in config_data.items():
        setattr(merged, key, value)

    for key, value in vars(cli_args).items():
        if value != getattr(default_args, key):
            setattr(merged, key, value)

    setattr(merged, "config", pre_args.config)
    return merged


def build_parser():
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
    )
    parser.add_argument(
        "--checkpoint_path", 
        default = "", 
        type = str, 
        help = "for reloading checkpoint and keep training"
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
    # model parameters
    parser.add_argument(
        "--model", 
        type = str, 
        default = 'FLEX', 
        help = "model"
    )    
    # U-Net parameters
    parser.add_argument(
        "--base_width", 
        type = int, 
        default = 128, 
        help = "Basewidth of U-Net"
    )
    return parser


def ddp_setup(local_rank, world_size):
    """
    Args:
        rank: Unique identifixer of each process
        world_size: Total number of processes
    """
    
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "3522"
        
        init_process_group(
            backend = "gloo", # nccl for multi-gpu, gloo for single-gpu
            rank = local_rank, 
            world_size = world_size
        )
        rank = local_rank
        
    else:
        init_process_group(
            backend="gloo", 
            init_method='env://'
        )
        #overwrite variables with correct values from env
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = get_rank()

    torch.cuda.set_device(local_rank)
    torch.backends.cudnn.benchmark = True
    return local_rank, rank

class Trainer:
    def __init__(
            self,
            # model: torch.nn.Module,
            flowComp: torch.nn.Module,
            ArbTimeFlowIntrp: torch.nn.Module, 
            trainFlowBackWarp: None,
            train_loader: DataLoader,
            optimizer: torch.optim.Optimizer,
            gpu_id: int,
            local_gpu_id: int,
            sampling_freq: int,
            run: wandb,
            run_name: str
        ) -> None:

        self.gpu_id = gpu_id
        self.local_gpu_id = local_gpu_id
        # self.model = model.to(local_gpu_id)
        self.flowComp = flowComp.to(local_gpu_id)
        self.ArbTimeFlowIntrp = ArbTimeFlowIntrp.to(local_gpu_id)   
        self.trainFlowBackWarp = trainFlowBackWarp.to(local_gpu_id)
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.sampling_freq = sampling_freq
        # self.model = DDP(
        #     model, 
        #     device_ids = [local_gpu_id], 
        #     find_unused_parameters = True
        # )
        self.flowComp = DDP(
            flowComp, 
            device_ids = [local_gpu_id], 
            find_unused_parameters = False
        )
        self.ArbTimeFlowIntrp = DDP(
            ArbTimeFlowIntrp, 
            device_ids = [local_gpu_id], 
            find_unused_parameters = False
        )

        self.run = run
        self.run_name = run_name
    
    # def _run_batch(
    #         self, 
    #         targets, 
    #         condition_start, 
    #         condition_end,  
    #         reynolds_number,
    #         target_interp_step,
    #         total_interp_steps
    #     ):
    #     self.optimizer.zero_grad()
    #     # if isinstance(self.model.module, DiffusionModel):
    #     #     reynolds_number = reynolds_number.unsqueeze(-1)
        
    #     loss = self.model(
    #         targets, 
    #         condition_start, 
    #         condition_end, 
    #         reynolds_number, 
    #         target_interp_step,
    #         total_interp_steps
    #     )
        
    #     loss.backward()
    #     if isinstance(self.model.module, DiffusionModel):
    #         torch.nn.utils.clip_grad_norm_(self.model.module.parameters(), 1.)
    #     self.optimizer.step()
    #     self.lr_scheduler.step()
    #     # self.model.module.ema.update()
    #     return loss.item()

    def _run_epoch(
            self, 
            epoch,
            L1_lossFn, 
            MSE_LossFn
        ):        
        self.train_loader.sampler.set_epoch(epoch)
        loss_values_task = []
        for inputs, targets, cond_params in self.train_loader:
            # Unpack the input tuple
            # condition_start, condition_end = inputs
            # condition_start = condition_start.to(self.local_gpu_id)
            # condition_end = condition_end.to(self.local_gpu_id)
            # targets = targets.to(self.local_gpu_id)
            I0, I1 = inputs
            I0 = I0.to(self.local_gpu_id)
            I1 = I1.to(self.local_gpu_id)
            IFrame = targets.to(self.local_gpu_id)

            # unpack the condition parameters
            target_interp_step, total_interp_steps, reynolds_number = cond_params
            #reynolds_number = reynolds_number.to(self.local_gpu_id)
            target_interp_step = target_interp_step.to(self.local_gpu_id)
            #total_interp_steps = total_interp_steps.to(self.local_gpu_id)

            trainFrameIndex = target_interp_step

            self.optimizer.zero_grad()
            
            # --- START Super Slomo ---
            # Calculate flow between reference frames I0 and I1
            # c: 2 -> 4 (was 6 -> 4)
            flowOut = self.flowComp(
                torch.cat((I0, I1), dim=1)
            )

            # Extracting flows between I0 and I1 - F_0_1 and F_1_0
            F_0_1 = flowOut[:,:2,:,:] # c = 2
            F_1_0 = flowOut[:,2:,:,:] # c = 2

            fCoeff = slomo_model.getFlowCoeff(trainFrameIndex, self.local_gpu_id)
            
            # Calculate intermediate flows
            # c = 2
            F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
            F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0

            # Get intermediate frames from the intermediate flows
            # c: (img + flow) -> img, (1 + 2) -> 1 (was (3 + 2) -> 3)
            g_I0_F_t_0 = self.trainFlowBackWarp(I0, F_t_0)
            g_I1_F_t_1 = self.trainFlowBackWarp(I1, F_t_1)

            # Calculate optical flow residuals and visibility maps
            # 2 flows + 1 visibility = (2*2 + 1) channels
            # c: 12 -> 5 (was 20 -> 5)
            intrpOut = self.ArbTimeFlowIntrp(
                torch.cat(
                    (
                        I0, 
                        I1, 
                        F_0_1, 
                        F_1_0, 
                        F_t_1, 
                        F_t_0, 
                        g_I1_F_t_1, 
                        g_I0_F_t_0
                    ), 
                    dim = 1
                )
            )

            # Extract optical flow residuals and visibility maps
            F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
            F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
            V_t_0   = F.sigmoid(intrpOut[:, 4:5, :, :])
            V_t_1   = 1 - V_t_0
        
            # Get intermediate frames from the intermediate flows
            g_I0_F_t_0_f = self.trainFlowBackWarp(I0, F_t_0_f)
            g_I1_F_t_1_f = self.trainFlowBackWarp(I1, F_t_1_f)
        
            wCoeff = slomo_model.getWarpCoeff(trainFrameIndex, self.local_gpu_id)
        
            # Calculate final intermediate frame 
            Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / (wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1)
        
            # Loss
            recnLoss = L1_lossFn(Ft_p, IFrame)
            # prcpLoss = MSE_LossFn(vgg16_conv_4_3(Ft_p), vgg16_conv_4_3(IFrame))
            warpLoss = L1_lossFn(g_I0_F_t_0, IFrame) + L1_lossFn(g_I1_F_t_1, IFrame) + L1_lossFn(self.trainFlowBackWarp(I0, F_1_0), I1) + L1_lossFn(self.trainFlowBackWarp(I1, F_0_1), I0)
        
            loss_smooth_1_0 = torch.mean(torch.abs(F_1_0[:, :, :, :-1] - F_1_0[:, :, :, 1:])) + torch.mean(torch.abs(F_1_0[:, :, :-1, :] - F_1_0[:, :, 1:, :]))
            loss_smooth_0_1 = torch.mean(torch.abs(F_0_1[:, :, :, :-1] - F_0_1[:, :, :, 1:])) + torch.mean(torch.abs(F_0_1[:, :, :-1, :] - F_0_1[:, :, 1:, :]))
            loss_smooth = loss_smooth_1_0 + loss_smooth_0_1
          
            # Total Loss - Coefficients 204 and 102 are used instead of 0.8 and 0.4
            # since the loss in paper is calculated for input pixels in range 0-255
            # and the input to our network is in range 0-1
            # loss = 204 * recnLoss + 102 * warpLoss + 0.005 * prcpLoss + loss_smooth

            # also, drop vgg loss due to only 1 channel input
            loss_task = 0.8 * recnLoss + 0.4 * warpLoss + loss_smooth

            # loss_task = self._run_batch(
            #     targets, 
            #     condition_start, 
            #     condition_end, 
            #     reynolds_number, 
            #     target_interp_step,
            #     total_interp_steps
            # )
            loss_task.backward()
            loss_values_task.append(loss_task.item())#loss.item()
            
            # if isinstance(self.model.module, DiffusionModel):
            #     torch.nn.utils.clip_grad_norm_(self.model.module.parameters(), 1.)
            self.optimizer.step()
        self.lr_scheduler.step() # super slomo does not need per-step lr update
        # --- END Super Slomo ---

        self.run.log({"Task loss": np.mean(loss_values_task)})
        # self.run.log({"Contrastive loss": 0})  
        return loss_values_task

    def _generate_samples(self, epoch):
        sample_dir = "./samples"
        os.makedirs(sample_dir, exist_ok = True)

        sample_path = "{}/train_samples_{}".format(
            sample_dir,
            self.run_name
        )
        os.makedirs(sample_path, exist_ok = True)

        # with self.model.module.ema.average_parameters():
        # self.model.eval()
        self.flowComp.eval()
        self.ArbTimeFlowIntrp.eval()

        with torch.no_grad():
            self.train_loader.sampler.set_epoch(1)

            # unpack the data
            inputs, targets, cond_params = next(iter(self.train_loader))
            # condition_start, condition_end = inputs
            # condition_start = condition_start.to(self.local_gpu_id)
            # condition_end = condition_end.to(self.local_gpu_id)
            
            # # unpack the condition parameters
            # target_interp_step, total_interp_steps, reynolds_number = cond_params
            # reynolds_number = reynolds_number.to(self.local_gpu_id)
            # target_interp_step = target_interp_step.to(self.local_gpu_id)
            # total_interp_steps = total_interp_steps.to(self.local_gpu_id)
            # # print(f'Type {type(target_interp_step)}')

            # # if isinstance(self.model.module, DiffusionModel):
            # #     reynolds_number = reynolds_number.unsqueeze(-1)
            
            # samples = self.model.module.sample(
            #     condition_start,
            #     condition_end,
            #     reynolds_number,
            #     target_interp_step,
            #     total_interp_steps
            # )

            I0, I1 = inputs
            I0 = I0.to(self.local_gpu_id)
            I1 = I1.to(self.local_gpu_id)
            IFrame = targets.to(self.local_gpu_id)

            # unpack the condition parameters
            target_interp_step, total_interp_steps, reynolds_number = cond_params
            #reynolds_number = reynolds_number.to(self.local_gpu_id)
            target_interp_step = target_interp_step.to(self.local_gpu_id)
            #total_interp_steps = total_interp_steps.to(self.local_gpu_id)

            trainFrameIndex = target_interp_step
            
            # --- START Super Slomo ---
            # Calculate flow between reference frames I0 and I1
            # c: 2 -> 4 (was 6 -> 4)
            flowOut = self.flowComp(
                torch.cat((I0, I1), dim=1)
            )

            # Extracting flows between I0 and I1 - F_0_1 and F_1_0
            F_0_1 = flowOut[:,:2,:,:] # c = 2
            F_1_0 = flowOut[:,2:,:,:] # c = 2

            fCoeff = slomo_model.getFlowCoeff(trainFrameIndex, self.local_gpu_id)
            
            # Calculate intermediate flows
            # c = 2
            F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
            F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0

            # Get intermediate frames from the intermediate flows
            # c: (img + flow) -> img, (1 + 2) -> 1 (was (3 + 2) -> 3)
            g_I0_F_t_0 = self.trainFlowBackWarp(I0, F_t_0)
            g_I1_F_t_1 = self.trainFlowBackWarp(I1, F_t_1)

            # Calculate optical flow residuals and visibility maps
            # 2 flows + 1 visibility = (2*2 + 1) channels
            # c: 12 -> 5 (was 20 -> 5)
            intrpOut = self.ArbTimeFlowIntrp(
                torch.cat(
                    (
                        I0, 
                        I1, 
                        F_0_1, 
                        F_1_0, 
                        F_t_1, 
                        F_t_0, 
                        g_I1_F_t_1, 
                        g_I0_F_t_0
                    ), 
                    dim = 1
                )
            )

            # Extract optical flow residuals and visibility maps
            F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
            F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
            V_t_0   = F.sigmoid(intrpOut[:, 4:5, :, :])
            V_t_1   = 1 - V_t_0
        
            # Get intermediate frames from the intermediate flows
            g_I0_F_t_0_f = self.trainFlowBackWarp(I0, F_t_0_f)
            g_I1_F_t_1_f = self.trainFlowBackWarp(I1, F_t_1_f)
        
            wCoeff = slomo_model.getWarpCoeff(trainFrameIndex, self.local_gpu_id)
        
            # Calculate final intermediate frame 
            Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / (wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1)
        
            # define the parameters
            samples = Ft_p
            condition_start = I0
            targets = IFrame

        plot_samples(samples, condition_start, targets, sample_path, epoch)
        print(f"Epoch {epoch} | Generated samples saved at {sample_path}")

    def _save_checkpoint(self, epoch, name=''):
        checkpoint_dir = "./checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)

        suffix = name or ""
        save_path = "{}/checkpoint_{}{}.pt".format(
            checkpoint_dir,
            self.run_name,
            suffix,
        )
        save_dict = {
            # 'model': self.model.module.state_dict(),
            'flowComp': self.flowComp.module.state_dict(),
            'ArbTimeFlowIntrp': self.ArbTimeFlowIntrp.module.state_dict(),
            # 'ema': self.model.module.ema.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch,
        }
        torch.save(save_dict, save_path)
        
        if name == '':
            print(f"Epoch {epoch} | Training checkpoint saved at {save_path}")

    def train(
            self, 
            max_epochs,
            L1_lossFn, 
            MSE_LossFn
        ):
        print('--- Starting training ---')
        self.lr_scheduler = scheduler(
            optimizer = self.optimizer,
            num_warmup_steps = len(self.train_loader) * 3, # short warmup phase
            num_training_steps = (len(self.train_loader) * max_epochs)
        )
        best_mse = np.inf

        # self.model.train()
        self.flowComp.train()
        self.ArbTimeFlowIntrp.train()

        for epoch in range(max_epochs):
            loss_values = self._run_epoch(
                epoch,
                L1_lossFn, 
                MSE_LossFn
            )

            if self.local_gpu_id == 0:
                avg_loss = np.mean(loss_values)
                print("Epoch {} | loss {:.4f} | learning rate {:.6f}".format(
                    epoch + 1, 
                    avg_loss, 
                    self.lr_scheduler.get_last_lr()[0]
                ))
                self.run.log({"loss": avg_loss})

                # Save the last and best checkpoint
                self._save_checkpoint(epoch + 1, name = '_last')
                
                if best_mse > avg_loss:
                    self._save_checkpoint(epoch + 1, name='_best')
                    best_mse = avg_loss

                # Generate samples at specified intervals
                if (
                    epoch == 0 or 
                    (epoch + 1) % self.sampling_freq == 0 or 
                    (epoch + 1) == max_epochs
                ):
                    self._generate_samples(epoch+1)
            
def load_checkpoint(
        save_path, 
        flowComp, 
        ArbTimeFlowIntrp, 
        optimizer, 
        device
    ):
    if not os.path.exists(save_path):
        print(f"Unable to load from {save_path}")

    checkpoint = torch.load(save_path, weights_only=True)
    flowComp.load_state_dict(checkpoint["flowComp"]) 
    ArbTimeFlowIntrp.load_state_dict(checkpoint["ArbTimeFlowIntrp"]) 
    # model.load_state_dict(checkpoint["model"])
    # ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
    # ema.load_state_dict(checkpoint["ema"])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print(f"Loaded model from {save_path}")

    return flowComp, ArbTimeFlowIntrp, optimizer
          
def load_train_objs(args, device):
    
    # load training set
    if args.data_name == "nskt":
        train_set = NSKT(
            patch_size = args.patch_size, 
            stride = args.stride,
            num_interp_steps= args.total_interp_steps_train,
            scratch_dir = args.scratch_dir,
            flag = "train",
            is_T_fixed = args.is_T_fixed
        )
    elif args.data_name == "shanghai":
        train_set = Shanghai(
            data_path = args.scratch_dir,
            img_size = args.patch_size, 
            type = "train",
            trans = None,
            total_interp_steps = args.total_interp_steps_train
        )
    elif args.data_name == "sea_temp":
        train_input_param = {
            'path': args.scratch_dir,
            'total_length': args.total_interp_steps_train, # total length of each sample (input + output)
            'input_length': 2, # length of input sequence
            'type': 'train', # train/test/valid
            'input_data_type': 'float32'
        }
        train_set = InputHandle(train_input_param)
    else:
        print("This dataset is not supported. We currently only support (nskt), (shanghai), and (sea_temp) datasets.")
        sys.exit()

    ema = None # placeholder for non-FLEX model
    if args.model == 'SuperSloMo':
        # model = SuperSloMo(
        #     device = 'cuda',
        #     time_step = args.time_steps
        # )
        # Initialize flow computation and arbitrary-time flow interpolation CNNs
        # original is (6, 4) 6 = 3 * 2 due to RGB input
        # for my case, 1 * 2 = 2 due to grayscale input
        flowComp = slomo_model.UNet(2, 4) 
        # flowComp.to(device) to device in main training loop

        # orginally (20, 5) due to RGB input
        # for my case, (1+2) * 4 = 12, 2 + 2 + 1 = 5
        ArbTimeFlowIntrp = slomo_model.UNet(12, 5) 
        # ArbTimeFlowIntrp.to(device)

        # Initialze backward warpers for train and validation datasets
        trainFlowBackWarp = slomo_model.backWarp(
            args.patch_size, 
            args.patch_size, 
            device
        )
        params = list(ArbTimeFlowIntrp.parameters()) + list(flowComp.parameters())
        # trainFlowBackWarp      = trainFlowBackWarp.to(device)
        # validationFlowBackWarp = model.backWarp(640, 352, device)
        # validationFlowBackWarp = validationFlowBackWarp.to(device)
    else:
        print("This model is not supported.")
        sys.exit()

    if args.optimizer == 'adam':
        # optimizer = torch.optim.Adam(
        #     model.parameters(), 
        #     lr = args.learning_rate
        # )
        optimizer = torch.optim.Adam(
            params, 
            lr = args.learning_rate
        )
    elif args.optimizer == 'lion':
        # optimizer = Lion(
        #     model.parameters(), 
        #     lr = args.learning_rate
        # )
        optimizer = Lion(
            params, 
            lr = args.learning_rate
        )
    else:
        print("Only Adam and Lion are supported.")
        sys.exit()
    return train_set, flowComp, ArbTimeFlowIntrp, trainFlowBackWarp, optimizer, ema

def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size = batch_size,
        pin_memory = True,
        shuffle = False,
        sampler = DistributedSampler(dataset),
        num_workers = 8,
        drop_last = True
    )

def main(
        rank: int, 
        world_size: int, 
        sampling_freq: int, 
        epochs: int, 
        batch_size: int, 
        run, 
        args
    ):
    
    local_rank, rank = ddp_setup(rank, world_size)
    print("local rank, rank: ", local_rank, rank)
    
    device = torch.cuda.current_device()
    print("device: ", device)
    slomo_model.configure_time_grid(args.data_name, args.total_interp_steps_train)
    
    dataset, flowComp, ArbTimeFlowIntrp, trainFlowBackWarp, optimizer, _ = load_train_objs(
        args = args, 
        device = device
    )
    # dataset, model, optimizer, ema = load_train_objs(args = args, device = device)
    train_data = prepare_dataloader(dataset, batch_size)

    # if ema is not None:
    #     model.ema = ema
    # torch.cuda.set_device(device)
    # model = model.to(device)

    flowComp = flowComp.to(device)
    ArbTimeFlowIntrp = ArbTimeFlowIntrp.to(device)
    trainFlowBackWarp = trainFlowBackWarp.to(device)

    # post-training checkpoint loading
    if args.checkpoint_path != '':
        flowComp, ArbTimeFlowIntrp, optimizer = load_checkpoint(
            args.checkpoint_path, 
            flowComp, 
            ArbTimeFlowIntrp,
            optimizer, 
            device
        )

    # define loss functions
    L1_lossFn = torch.nn.L1Loss()
    MSE_LossFn = torch.nn.MSELoss()
    # vgg16 = torchvision.models.vgg16(pretrained=True)
    # vgg16_conv_4_3 = torch.nn.Sequential(*list(vgg16.children())[0][:22])
    # vgg16_conv_4_3.to(device)
    # for param in vgg16_conv_4_3.parameters():
    #     param.requires_grad = False

    
    # Model summary
    print("**************")
    print("Total model params in the 1st UNet: %.2fM" % (
            sum(p.numel() for p in flowComp.parameters()) / 1000000.0
        )
    )
    print("Total model params in the 2nd UNet: %.2fM" % (
            sum(p.numel() for p in ArbTimeFlowIntrp.parameters()) / 1000000.0
        )
    )
    print("**************")
    
    # gpu_id: int, local_gpu_id: int,
    start = time.time()
    trainer = Trainer(
        flowComp,
        ArbTimeFlowIntrp, 
        trainFlowBackWarp,
        train_data, 
        optimizer, 
        gpu_id = rank, 
        local_gpu_id = local_rank, 
        sampling_freq = sampling_freq, 
        run = run, 
        run_name = args.run_name
    )
    trainer.train(epochs, L1_lossFn, MSE_LossFn)
    end = time.time()
    print("Training time: ", end - start)
    destroy_process_group()

if __name__ == "__main__":
    args = get_args_with_yaml()

    # Launch processes.
    print('Launching processes...')
    
    # wandb.login()
    wandb.login(key = "5282eaefee2cb8f881265effb6251abf1703deee")
    args.run_name = get_run_name(args)
    run = wandb.init(
        # Set the project where this run will be logged
        project = "InterpDM",
        name = args.run_name,
        # Track hyperparameters and run metadata
        config = {
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "batch size": args.batch_size,
            "total_interp_steps": args.total_interp_steps_train
        },
    )

    world_size = torch.cuda.device_count()
    print("world size: ", world_size)
    
    if world_size == 1:
        main(
            0, 
            world_size, 
            args.sampling_freq, 
            args.epochs, 
            args.batch_size, 
            run, 
            args
        )
    else:
        mp.spawn(
            main, 
            args = (
                world_size, 
                args.sampling_freq, 
                args.epochs, 
                args.batch_size, 
                run, 
                args
            ), 
            nprocs = world_size
        )
