import argparse
import os, sys, time
import numpy as np
import wandb
import torch
import yaml
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, get_rank
from diffusers.optimization import get_linear_schedule_with_warmup as scheduler
from src.plotting import plot_samples
from utils.params import get_args
from src.utilities import *

import warnings
warnings.filterwarnings("ignore")


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
            model: torch.nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            optimizer: torch.optim.Optimizer,
            gpu_id: int,
            local_gpu_id: int,
            sampling_freq: int,
            run: wandb,
            run_name: str
        ) -> None:
        
        self.gpu_id = gpu_id
        self.local_gpu_id = local_gpu_id
        self.model = model.to(local_gpu_id)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.sampling_freq = sampling_freq
        self.model = DDP(
            model, 
            device_ids = [local_gpu_id], 
            find_unused_parameters = True
        )
        self.run = run
        self.run_name = run_name
    
    def _run_batch(
            self, 
            targets, 
            condition_start, 
            condition_end,  
            reynolds_number,
            target_interp_step,
            total_interp_steps
        ):
        self.optimizer.zero_grad()

        loss = self.model(
            targets, 
            condition_start, 
            condition_end, 
            reynolds_number, 
            target_interp_step,
            total_interp_steps
        )
        
        loss.backward()
        if hasattr(self.model.module, "ema"):
            torch.nn.utils.clip_grad_norm_(self.model.module.parameters(), 1.)
        self.optimizer.step()
        self.lr_scheduler.step()
        return loss.item()

    def _run_epoch(self, epoch):        
        self.train_loader.sampler.set_epoch(epoch)
        loss_values_task = []
        for inputs, targets, cond_params in self.train_loader:
            # Unpack the input tuple
            condition_start, condition_end = inputs
            condition_start = condition_start.to(self.local_gpu_id)
            condition_end = condition_end.to(self.local_gpu_id)
            targets = targets.to(self.local_gpu_id)
            
            # unpack the condition parameters
            target_interp_step, total_interp_steps, reynolds_number = cond_params
            reynolds_number = reynolds_number.to(self.local_gpu_id)
            target_interp_step = target_interp_step.to(self.local_gpu_id)
            total_interp_steps = total_interp_steps.to(self.local_gpu_id)
            
            loss_task = self._run_batch(
                targets, 
                condition_start, 
                condition_end, 
                reynolds_number, 
                target_interp_step,
                total_interp_steps
            )
            loss_values_task.append(loss_task)

        self.run.log({"Task loss": np.mean(loss_values_task)})
        # self.run.log({"Contrastive loss": 0})  
        return loss_values_task

    def val_epoch(self, epoch):        
        self.val_loader.sampler.set_epoch(epoch)
        loss_values_task = []
        
        with self.model.module.ema.average_parameters():
            with torch.no_grad():
                self.model.eval()

            for inputs, targets, cond_params in self.val_loader:
                # Unpack the input tuple
                condition_start, condition_end = inputs
                condition_start = condition_start.to(self.local_gpu_id)
                condition_end = condition_end.to(self.local_gpu_id)
                targets = targets.to(self.local_gpu_id)
                
                # Unpack the condition parameters
                target_interp_step, total_interp_steps, reynolds_number = cond_params
                reynolds_number = reynolds_number.to(self.local_gpu_id)
                target_interp_step = target_interp_step.to(self.local_gpu_id)
                total_interp_steps = total_interp_steps.to(self.local_gpu_id)

                # predict
                val_loss = self.model(
                    targets, 
                    condition_start, 
                    condition_end, 
                    reynolds_number, 
                    target_interp_step,
                    total_interp_steps
                )
                loss_values_task.append(val_loss.item())
        self.run.log({"Val loss": np.mean(loss_values_task)})

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
        self.model.eval()
        with torch.no_grad():
            self.train_loader.sampler.set_epoch(1)

            # unpack the data
            inputs, targets, cond_params = next(iter(self.train_loader))
            condition_start, condition_end = inputs
            condition_start = condition_start.to(self.local_gpu_id)
            condition_end = condition_end.to(self.local_gpu_id)
            
            # unpack the condition parameters
            target_interp_step, total_interp_steps, reynolds_number = cond_params
            reynolds_number = reynolds_number.to(self.local_gpu_id)
            target_interp_step = target_interp_step.to(self.local_gpu_id)
            total_interp_steps = total_interp_steps.to(self.local_gpu_id)
            # print(f'Type {type(target_interp_step)}')

            # if isinstance(self.model.module, DiffusionModel):
            #     reynolds_number = reynolds_number.unsqueeze(-1)
            
            samples = self.model.module.sample(
                condition_start,
                condition_end,
                reynolds_number,
                target_interp_step,
                total_interp_steps
            )
        plot_samples(samples, condition_start, targets, sample_path, epoch)
        print(f"Epoch {epoch} | Generated samples saved at {sample_path}")

    def _save_checkpoint(self, epoch, name=''):
        checkpoint_dir = "./checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)

        save_path = "{}/checkpoint_{}.pt".format(
            checkpoint_dir,
            self.run_name
        )
        save_dict = {
            'model': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(save_dict, save_path)
        
        if name == '':
            print(f"Epoch {epoch} | Training checkpoint saved at {save_path}")

    def train(self, max_epochs: int):
        print('--- Starting training ---')
        self.lr_scheduler = scheduler(
            optimizer = self.optimizer,
            num_warmup_steps = len(self.train_loader) * 3, # short warmup phase
            num_training_steps = (len(self.train_loader) * max_epochs)
        )
        best_mse = np.inf
        self.model.train()
        for epoch in range(max_epochs):
            train_loss_values = self._run_epoch(epoch)
            val_loss_values = train_loss_values
            # val_loss_values = self.val_epoch(epoch)

            if self.local_gpu_id == 0:
                avg_train_loss = np.mean(train_loss_values)
                avg_val_loss = np.mean(val_loss_values)
                print("Epoch {} | Train loss {:.4f} | Val loss {:.4f} | learning rate {:.6f}".format(
                    epoch + 1, 
                    avg_train_loss, 
                    avg_val_loss,
                    self.lr_scheduler.get_last_lr()[0]
                ))
                # self.run.log({"loss": avg_loss})
                self.run.log({"Train loss": avg_train_loss})
                self.run.log({"Val loss": avg_val_loss})

                # Save the last and best checkpoint
                self._save_checkpoint(epoch + 1, name = '_last')
                
                if best_mse > avg_val_loss:
                    self._save_checkpoint(epoch + 1, name='_best')
                    best_mse = avg_val_loss

                # Generate samples at specified intervals
                if (
                    epoch == 0 or 
                    (epoch + 1) % self.sampling_freq == 0 or 
                    (epoch + 1) == max_epochs
                ):
                    self._generate_samples(epoch+1)

def load_checkpoint_unet(
        save_path, 
        model, 
        optimizer, 
        device
    ):
    if not os.path.exists(save_path):
        print(f"Unable to load from {save_path}")

    checkpoint = torch.load(save_path, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint['optimizer'])

    print(f"Loaded model from {save_path}")
    return model, optimizer


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
    
    train_set, val_set, model, optimizer, ema = load_train_objs(args = args)
    # train_loader, val_loader = prepare_dataloader(dataset, batch_size)
    train_loader = prepare_dataloader(train_set, batch_size)
    val_loader = prepare_dataloader(val_set, batch_size)

    if ema is not None:
        model.ema = ema
    # torch.cuda.set_device(device)
    model = model.to(device)

    # post-training checkpoint loading
    if args.checkpoint_path != '':
        model, optimizer = load_checkpoint_unet(
            args.checkpoint_path, 
            model, 
            optimizer, 
            device
        )
    
    # Model summary
    print("**************")
    print("Total model params: %.2fM" % (
            sum(p.numel() for p in model.parameters()) / 1000000.0
        )
    )
    print("**************")
    
    # gpu_id: int, local_gpu_id: int,
    start = time.time()
    trainer = Trainer(
        model, 
        train_loader, 
        val_loader,
        optimizer, 
        gpu_id = rank, 
        local_gpu_id = local_rank, 
        sampling_freq = sampling_freq, 
        run = run, 
        run_name = args.run_name
    )
    trainer.train(epochs)
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
