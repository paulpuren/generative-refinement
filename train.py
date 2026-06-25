import argparse
import json
import os, sys, time
import numpy as np
import wandb
import torch
import yaml
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, barrier, get_rank, is_initialized, all_reduce, get_world_size
# from diffusers.optimization import get_linear_schedule_with_warmup as scheduler
from diffusers.optimization import get_cosine_schedule_with_warmup as scheduler
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
    
    backend = "nccl" if torch.cuda.is_available() else "gloo"

    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "3522"
        
        init_process_group(
            backend = backend,
            rank = local_rank, 
            world_size = world_size
        )
        rank = local_rank
        
    else:
        init_process_group(
            backend = backend, 
            init_method='env://'
        )
        #overwrite variables with correct values from env
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = get_rank()

    torch.cuda.set_device(local_rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
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
            validation_freq: int,
            run: wandb,
            run_name: str,
            checkpoint_path: str = "",
            ddp_find_unused_parameters: bool = False,
        ) -> None:
        
        self.gpu_id = gpu_id
        self.local_gpu_id = local_gpu_id
        self.model = model.to(local_gpu_id)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.sampling_freq = sampling_freq
        self.validation_freq = max(1, int(validation_freq))
        self.checkpoint_path = checkpoint_path
        self.model = DDP(
            model, 
            device_ids = [local_gpu_id], 
            find_unused_parameters = ddp_find_unused_parameters
        )
        self.run = run
        self.run_name = run_name
        self.progress_path = os.path.join("./checkpoints", f"progress_{self.run_name}.json")

    def _write_progress(self, epoch, max_epochs, completed=False):
        progress = {
            "run_name": self.run_name,
            "epoch": int(epoch),
            "max_epochs": int(max_epochs),
            "completed": bool(completed),
        }
        with open(self.progress_path, "w", encoding="utf-8") as f:
            json.dump(progress, f, indent=2, sort_keys=True)
    
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
        # if isinstance(self.model.module, DiffusionModel):
        #     reynolds_number = reynolds_number.unsqueeze(-1)
        loss = self.model(
            targets, 
            condition_start, 
            condition_end, 
            reynolds_number, 
            target_interp_step,
            total_interp_steps
        )
        loss.backward()
        if isinstance(self.model.module, (DiffusionModel, DiffusionModelResidual)):
            torch.nn.utils.clip_grad_norm_(self.model.module.parameters(), 1.)
        self.optimizer.step()
        self.lr_scheduler.step()
        self.model.module.ema.update()
        return loss.item()

    def _run_epoch(self, epoch):        
        self.model.train()
        self.train_loader.sampler.set_epoch(epoch)
        loss_values_task = []
        for inputs, targets, cond_params in self.train_loader:
            # Unpack the input tuple
            condition_start, condition_end = inputs
            condition_start = condition_start.to(self.local_gpu_id, non_blocking=True)
            condition_end = condition_end.to(self.local_gpu_id, non_blocking=True)
            targets = targets.to(self.local_gpu_id, non_blocking=True)
            
            # Unpack the condition parameters
            target_interp_step, total_interp_steps, reynolds_number = cond_params
            reynolds_number = reynolds_number.to(self.local_gpu_id, non_blocking=True)
            target_interp_step = target_interp_step.to(self.local_gpu_id, non_blocking=True)
            total_interp_steps = total_interp_steps.to(self.local_gpu_id, non_blocking=True)
            
            loss_task = self._run_batch(
                targets, 
                condition_start, 
                condition_end, 
                reynolds_number, 
                target_interp_step,
                total_interp_steps
            )
            loss_values_task.append(loss_task)
        if self.local_gpu_id == 0 and self.run is not None:
            self.run.log({"Train loss": np.mean(loss_values_task)})

        return loss_values_task

    def val_epoch(self, epoch):        
        self.val_loader.sampler.set_epoch(epoch)
        loss_values_task = []
        
        with self.model.module.ema.average_parameters():
            self.model.eval()
            with torch.no_grad():
                for inputs, targets, cond_params in self.val_loader:
                    # Unpack the input tuple
                    condition_start, condition_end = inputs
                    condition_start = condition_start.to(self.local_gpu_id, non_blocking=True)
                    condition_end = condition_end.to(self.local_gpu_id, non_blocking=True)
                    targets = targets.to(self.local_gpu_id, non_blocking=True)
                    
                    # Unpack the condition parameters
                    target_interp_step, total_interp_steps, reynolds_number = cond_params
                    reynolds_number = reynolds_number.to(self.local_gpu_id, non_blocking=True)
                    target_interp_step = target_interp_step.to(self.local_gpu_id, non_blocking=True)
                    total_interp_steps = total_interp_steps.to(self.local_gpu_id, non_blocking=True)

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
        if self.local_gpu_id == 0 and self.run is not None:
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

        with self.model.module.ema.average_parameters():

            self.model.eval()
            with torch.no_grad():
                self.train_loader.sampler.set_epoch(1)

                # unpack the data
                inputs, targets, cond_params = next(iter(self.train_loader))
                condition_start, condition_end = inputs
                condition_start = condition_start.to(self.local_gpu_id, non_blocking=True)
                condition_end = condition_end.to(self.local_gpu_id, non_blocking=True)
                
                # unpack the condition parameters
                target_interp_step, total_interp_steps, reynolds_number = cond_params
                reynolds_number = reynolds_number.to(self.local_gpu_id, non_blocking=True)
                target_interp_step = target_interp_step.to(self.local_gpu_id, non_blocking=True)
                total_interp_steps = total_interp_steps.to(self.local_gpu_id, non_blocking=True)
                # print(f'Type {type(target_interp_step)}')

                # if isinstance(self.model.module, DiffusionModel):
                #     reynolds_number = reynolds_number.unsqueeze(-1)
                
                samples = self.model.module.sample(
                    targets.shape[0],
                    (1, targets.shape[2], targets.shape[3]),
                    condition_start,
                    condition_end,
                    reynolds_number,
                    target_interp_step,
                    total_interp_steps,
                    self.local_gpu_id
                )
        plot_samples(samples, condition_start, targets, sample_path, epoch)
        print(f"Epoch {epoch} | Generated samples saved at {sample_path}")

    def _save_checkpoint(self, epoch, max_epochs, name='', completed=False):
        checkpoint_dir = "./checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)

        save_path = "{}/checkpoint_{}{}.pt".format(
            checkpoint_dir,
            self.run_name,
            name,
        ) if name else "{}/checkpoint_{}.pt".format(
            checkpoint_dir,
            self.run_name,
        )
        save_dict = {
            'model': self.model.module.state_dict(),
            'ema': self.model.module.ema.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': int(epoch),
        }
        torch.save(save_dict, save_path)
        self._write_progress(epoch=epoch, max_epochs=max_epochs, completed=completed)
        
        print(f"Epoch {epoch} | Training checkpoint saved at {save_path}")

    def train(self, max_epochs: int, start_epoch: int = 0):
        print('--- Starting training ---')
        if self.checkpoint_path == '':
            self.lr_scheduler = scheduler(
                optimizer = self.optimizer,
                num_warmup_steps = len(self.train_loader) * 3,
                num_training_steps = (len(self.train_loader) * max_epochs)
            )
        else:
            self.lr_scheduler = scheduler(
                optimizer = self.optimizer,
                num_warmup_steps = len(self.train_loader) * 0, 
                num_training_steps = (len(self.train_loader) * max_epochs)
            )

        if start_epoch > 0:
            self.lr_scheduler.step(start_epoch * len(self.train_loader))

        best_mse = np.inf
        self.model.train()

        if start_epoch >= max_epochs:
            if self.local_gpu_id == 0:
                self._write_progress(
                    epoch=max_epochs, 
                    max_epochs=max_epochs, 
                    completed=True
                )
                print(f"Run already complete at epoch {start_epoch}.")
            return
        
        for epoch in range(start_epoch, max_epochs):
            # train one epoch
            train_loss_values = self._run_epoch(epoch)
            run_validation = (
                epoch == 0 or
                (epoch + 1) % self.validation_freq == 0 or
                (epoch + 1) == max_epochs
            )
            avg_val_loss = None
            if run_validation:
                val_loss_values = self.val_epoch(epoch)
                avg_val_loss = np.mean(val_loss_values)

            if self.local_gpu_id == 0:
                avg_train_loss = np.mean(train_loss_values)
                if avg_val_loss is None:
                    print("Epoch {} | Train loss {:.4f} | Val loss skipped | learning rate {:.6f}".format(
                        epoch + 1,
                        avg_train_loss,
                        self.lr_scheduler.get_last_lr()[0]
                    ))
                else:
                    print("Epoch {} | Train loss {:.4f} | Val loss {:.4f} | learning rate {:.6f}".format(
                        epoch + 1, 
                        avg_train_loss, 
                        avg_val_loss,
                        self.lr_scheduler.get_last_lr()[0]
                    ))
                self.run.log({"Train loss": avg_train_loss})
                if avg_val_loss is not None:
                    self.run.log({"Val loss": avg_val_loss})

                # save the latest checkpoint
                self._save_checkpoint(
                    epoch+1, 
                    max_epochs=max_epochs, 
                    name='_last'
                )
                
                if avg_val_loss is not None and best_mse > avg_val_loss:
                    # save the best checkpoint
                    self._save_checkpoint(
                        epoch+1, 
                        max_epochs=max_epochs, 
                        name='_best'
                    )
                    best_mse = avg_val_loss

                # Generate samples at specified intervals
                if (
                    epoch == 0 or 
                    (epoch + 1) % self.sampling_freq == 0 or 
                    (epoch + 1) == max_epochs
                ):
                    self._generate_samples(epoch+1)

        if self.local_gpu_id == 0:
            self._write_progress(
                epoch=max_epochs, 
                max_epochs=max_epochs, 
                completed=True
            )

def main(
        rank: int, 
        world_size: int, 
        sampling_freq: int, 
        validation_freq: int,
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
    
    run = None
    if rank == 0:
        wandb.login(key = "5282eaefee2cb8f881265effb6251abf1703deee")
        run = wandb.init(
            project = "InterpDM",
            name = args.run_name,
            config = {
                "learning_rate": args.learning_rate,
                "epochs": args.epochs,
                "batch size": args.batch_size,
                "total_interp_steps": args.total_interp_steps_train
            },
        )

    train_loader = prepare_dataloader(
        train_set,
        batch_size,
        num_workers = args.num_workers,
        persistent_workers = args.persistent_workers,
        prefetch_factor = args.prefetch_factor,
    )
    val_loader = prepare_dataloader(
        val_set,
        batch_size,
        num_workers = args.num_workers,
        persistent_workers = args.persistent_workers,
        prefetch_factor = args.prefetch_factor,
    )

    if ema is not None:
        model.ema = ema
    model = model.to(device)

    # post-training checkpoint loading
    start_epoch = 0
    if args.checkpoint_path != '':
        model, ema, optimizer, start_epoch = load_checkpoint(
            args.checkpoint_path, 
            model, 
            optimizer, 
            device,
            load_optimizer_state=args.resume_training,
        )
        if not args.resume_training:
            for param_group in optimizer.param_groups:
                param_group["lr"] = args.learning_rate
                param_group["initial_lr"] = args.learning_rate
        print("Configured learning rate:", args.learning_rate)
        print("Optimizer learning rate after resume:", optimizer.param_groups[0]["lr"])
        print("Resume training:", args.resume_training)
        print("Starting from epoch:", start_epoch)
    
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
        validation_freq = validation_freq,
        run = run, 
        run_name = args.run_name,
        checkpoint_path = args.checkpoint_path,
        ddp_find_unused_parameters = args.ddp_find_unused_parameters,
    )
    trainer.train(epochs, start_epoch=start_epoch)
    end = time.time()
    print("Training time: ", end - start)
    if run is not None:
        run.finish()
    destroy_process_group()

if __name__ == "__main__":
    args = get_args_with_yaml()

    # Launch processes.
    print('Launching processes...')
    
    args.run_name = get_run_name(args)
    world_size = torch.cuda.device_count()
    print("world size: ", world_size)
    
    if world_size == 1:
        main(
            0, 
            world_size, 
            args.sampling_freq, 
            args.validation_freq,
            args.epochs, 
            args.batch_size, 
            None, 
            args
        )
    else:
        mp.spawn(
            main, 
            args = (
                world_size, 
                args.sampling_freq, 
                args.validation_freq,
                args.epochs, 
                args.batch_size, 
                None, 
                args
            ), 
            nprocs = world_size
        )
