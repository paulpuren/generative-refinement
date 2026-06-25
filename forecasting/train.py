import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch
import torch.multiprocessing as mp
import wandb
import yaml
from diffusers.optimization import get_cosine_schedule_with_warmup as scheduler
from torch.distributed import destroy_process_group, get_rank, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from forecasting.diffusion_model import ForecastingDiffusionModel
from forecasting.params import get_train_args
from forecasting.utilities import (
    get_run_name,
    load_checkpoint,
    load_train_objs,
    prepare_dataloader,
)


def _get_args_from_argv(argv):
    saved_argv = sys.argv
    try:
        sys.argv = argv
        return get_train_args()
    finally:
        sys.argv = saved_argv


def load_yaml_config(config_path):
    if not config_path:
        return {}
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
    merged.total_interp_steps_train = 1
    merged.is_T_fixed = True
    merged.config = pre_args.config
    return merged


def ddp_setup(local_rank, world_size):
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "3523"
        init_process_group(backend=backend, rank=local_rank, world_size=world_size)
        rank = local_rank
    else:
        init_process_group(backend=backend, init_method="env://")
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
        model,
        train_loader,
        val_loader,
        optimizer,
        local_gpu_id,
        sampling_freq,
        validation_freq,
        run,
        run_name,
        checkpoint_path="",
        ddp_find_unused_parameters=False,
        direct_horizon_loss_weight=1.0,
        rollout_loss_weight=0.0,
        forecast_train_horizon=1,
        rollout_batch_size=0,
    ):
        self.local_gpu_id = local_gpu_id
        self.model = DDP(
            model.to(local_gpu_id),
            device_ids=[local_gpu_id],
            find_unused_parameters=ddp_find_unused_parameters,
        )
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.sampling_freq = sampling_freq
        self.validation_freq = max(1, int(validation_freq))
        self.run = run
        self.run_name = run_name
        self.checkpoint_path = checkpoint_path
        self.progress_path = os.path.join("./checkpoints", f"progress_{self.run_name}.json")
        self.direct_horizon_loss_weight = float(direct_horizon_loss_weight)
        self.rollout_loss_weight = float(rollout_loss_weight)
        self.forecast_train_horizon = max(1, int(forecast_train_horizon))
        self.rollout_batch_size = max(0, int(rollout_batch_size))

    def _write_progress(self, epoch, max_epochs, completed=False):
        os.makedirs("./checkpoints", exist_ok=True)
        with open(self.progress_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "run_name": self.run_name,
                    "epoch": int(epoch),
                    "max_epochs": int(max_epochs),
                    "completed": bool(completed),
                },
                f,
                indent=2,
                sort_keys=True,
            )

    def _move_batch(self, inputs, targets, cond_params):
        condition_start, condition_end = inputs
        target_interp_step, total_interp_steps, reynolds_number = cond_params
        if isinstance(targets, (list, tuple)):
            moved_targets = [
                target.to(self.local_gpu_id, non_blocking=True) for target in targets
            ]
        else:
            moved_targets = targets.to(self.local_gpu_id, non_blocking=True)
        return (
            moved_targets,
            condition_start.to(self.local_gpu_id, non_blocking=True),
            condition_end.to(self.local_gpu_id, non_blocking=True),
            reynolds_number.to(self.local_gpu_id, non_blocking=True),
            target_interp_step.to(self.local_gpu_id, non_blocking=True),
            total_interp_steps.to(self.local_gpu_id, non_blocking=True),
        )

    @staticmethod
    def _target_sequence(targets):
        if isinstance(targets, (list, tuple)):
            return list(targets)
        return [targets]

    def _direct_horizon_loss(
        self,
        targets,
        condition_start,
        condition_end,
        reynolds_number,
    ):
        target_seq = self._target_sequence(targets)
        batch_size = condition_start.shape[0]
        horizon = len(target_seq)
        target_stack = torch.stack(target_seq, dim=1)
        lead_idx = torch.randint(
            0,
            horizon,
            (batch_size,),
            device=condition_start.device,
        )
        gather_shape = (batch_size, 1, *target_stack.shape[2:])
        selected_targets = target_stack.gather(
            1,
            lead_idx.view(batch_size, 1, 1, 1, 1).expand(gather_shape),
        ).squeeze(1)
        lead_steps = (lead_idx + 1).float()
        return self.model(
            selected_targets,
            condition_start,
            condition_end,
            reynolds_number,
            lead_steps,
            lead_steps,
        )

    def _rollout_loss(
        self,
        targets,
        condition_start,
        condition_end,
        reynolds_number,
    ):
        target_seq = self._target_sequence(targets)
        if self.rollout_batch_size and condition_start.shape[0] > self.rollout_batch_size:
            subset = torch.randperm(
                condition_start.shape[0],
                device=condition_start.device,
            )[: self.rollout_batch_size]
            condition_start = condition_start.index_select(0, subset)
            condition_end = condition_end.index_select(0, subset)
            reynolds_number = reynolds_number.index_select(0, subset)
            target_seq = [target.index_select(0, subset) for target in target_seq]
        batch_size = condition_start.shape[0]
        one_step = torch.ones(batch_size, dtype=torch.float32, device=condition_start.device)
        current = condition_start
        previous = condition_end
        losses = []
        for target in target_seq:
            prediction = self.model.module.sample_differentiable(
                batch_size,
                (1, target.shape[2], target.shape[3]),
                current,
                previous,
                reynolds_number,
                one_step,
                one_step,
                self.local_gpu_id,
            )
            losses.append(torch.nn.functional.l1_loss(prediction, target))
            previous = current
            current = prediction
        return torch.stack(losses).mean()

    def _run_batch(self, batch):
        inputs, targets, cond_params = batch
        targets, condition_start, condition_end, reynolds_number, _, _ = self._move_batch(
            inputs, targets, cond_params
        )
        self.optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        if self.direct_horizon_loss_weight:
            direct_loss = self.direct_horizon_loss_weight * self._direct_horizon_loss(
                targets,
                condition_start,
                condition_end,
                reynolds_number,
            )
            loss_value += direct_loss.item()
            direct_loss.backward()
            del direct_loss
        if self.rollout_loss_weight:
            rollout_loss = self.rollout_loss_weight * self._rollout_loss(
                targets,
                condition_start,
                condition_end,
                reynolds_number,
            )
            loss_value += rollout_loss.item()
            rollout_loss.backward()
            del rollout_loss
        if isinstance(self.model.module, ForecastingDiffusionModel):
            torch.nn.utils.clip_grad_norm_(self.model.module.parameters(), 1.0)
        self.optimizer.step()
        self.lr_scheduler.step()
        self.model.module.ema.update()
        return loss_value

    def _run_epoch(self, epoch):
        self.model.train()
        self.train_loader.sampler.set_epoch(epoch)
        return [self._run_batch(batch) for batch in self.train_loader]

    def val_epoch(self, epoch):
        self.val_loader.sampler.set_epoch(epoch)
        losses = []
        with self.model.module.ema.average_parameters():
            self.model.eval()
            with torch.no_grad():
                for inputs, targets, cond_params in self.val_loader:
                    (
                        targets,
                        condition_start,
                        condition_end,
                        reynolds_number,
                        _,
                        _,
                    ) = self._move_batch(inputs, targets, cond_params)
                    losses.append(
                        self._direct_horizon_loss(
                            targets,
                            condition_start,
                            condition_end,
                            reynolds_number,
                        ).item()
                    )
        return losses

    def _generate_samples(self, epoch):
        sample_dir = os.path.join("./samples", f"train_samples_{self.run_name}")
        os.makedirs(sample_dir, exist_ok=True)
        with self.model.module.ema.average_parameters():
            self.model.eval()
            with torch.no_grad():
                inputs, targets, cond_params = next(iter(self.train_loader))
                condition_start, condition_end = inputs
                target_interp_step, total_interp_steps, reynolds_number = cond_params
                condition_start = condition_start.to(self.local_gpu_id)
                condition_end = condition_end.to(self.local_gpu_id)
                reynolds_number = reynolds_number.to(self.local_gpu_id)
                target_seq = self._target_sequence(targets)
                sample_target = target_seq[0]
                target_interp_step = torch.ones(
                    condition_start.shape[0],
                    dtype=torch.float32,
                    device=self.local_gpu_id,
                )
                total_interp_steps = target_interp_step
                samples = self.model.module.sample(
                    sample_target.shape[0],
                    (1, sample_target.shape[2], sample_target.shape[3]),
                    condition_start,
                    condition_end,
                    reynolds_number,
                    target_interp_step,
                    total_interp_steps,
                    self.local_gpu_id,
                )
        np.save(
            os.path.join(sample_dir, f"one_step_sample_{epoch}.npy"),
            {
                "condition_start": condition_start.cpu().numpy(),
                "condition_prev": condition_end.cpu().numpy(),
                "targets": sample_target.numpy(),
                "predictions": samples.cpu().numpy(),
            },
        )

    def _save_checkpoint(self, epoch, max_epochs, name="", completed=False):
        os.makedirs("./checkpoints", exist_ok=True)
        suffix = name if name else ""
        save_path = os.path.join("./checkpoints", f"checkpoint_{self.run_name}{suffix}.pt")
        torch.save(
            {
                "model": self.model.module.state_dict(),
                "ema": self.model.module.ema.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch": int(epoch),
            },
            save_path,
        )
        self._write_progress(epoch, max_epochs, completed=completed)
        print(f"Epoch {epoch} | checkpoint saved at {save_path}")

    def train(self, max_epochs, start_epoch=0):
        self.lr_scheduler = scheduler(
            optimizer=self.optimizer,
            num_warmup_steps=len(self.train_loader) * (0 if self.checkpoint_path else 3),
            num_training_steps=len(self.train_loader) * max_epochs,
        )
        if start_epoch > 0:
            self.lr_scheduler.step(start_epoch * len(self.train_loader))

        best_val = np.inf
        for epoch in range(start_epoch, max_epochs):
            train_losses = self._run_epoch(epoch)
            run_validation = (
                epoch == 0
                or (epoch + 1) % self.validation_freq == 0
                or (epoch + 1) == max_epochs
            )
            avg_val = None
            if run_validation:
                avg_val = np.mean(self.val_epoch(epoch))

            if self.local_gpu_id == 0:
                avg_train = np.mean(train_losses)
                if self.run is not None:
                    self.run.log({"Train loss": avg_train})
                    if avg_val is not None:
                        self.run.log({"Val loss": avg_val})
                if avg_val is None:
                    print(f"Epoch {epoch + 1} | Train loss {avg_train:.4f} | Val skipped")
                else:
                    print(f"Epoch {epoch + 1} | Train loss {avg_train:.4f} | Val loss {avg_val:.4f}")
                self._save_checkpoint(epoch + 1, max_epochs, name="_last")
                if avg_val is not None and avg_val < best_val:
                    best_val = avg_val
                    self._save_checkpoint(epoch + 1, max_epochs, name="_best")
                if (
                    epoch == 0
                    or (epoch + 1) % self.sampling_freq == 0
                    or (epoch + 1) == max_epochs
                ):
                    self._generate_samples(epoch + 1)
        if self.local_gpu_id == 0:
            self._write_progress(max_epochs, max_epochs, completed=True)


def main(rank, world_size, args):
    local_rank, rank = ddp_setup(rank, world_size)
    device = torch.cuda.current_device()
    train_set, val_set, model, optimizer, ema = load_train_objs(args)

    run = None
    if rank == 0 and args.wandb:
        run = wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config=vars(args),
        )

    train_loader = prepare_dataloader(
        train_set,
        args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )
    val_loader = prepare_dataloader(
        val_set,
        args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )

    model.ema = ema
    start_epoch = 0
    if args.checkpoint_path:
        model, ema, optimizer, start_epoch = load_checkpoint(
            args.checkpoint_path,
            model,
            optimizer,
            device,
            load_optimizer_state=args.resume_training,
        )
        model.ema = ema
        if not args.resume_training:
            for param_group in optimizer.param_groups:
                param_group["lr"] = args.learning_rate
                param_group["initial_lr"] = args.learning_rate

    print("Total model params: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1e6))
    start = time.time()
    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        optimizer,
        local_rank,
        args.sampling_freq,
        args.validation_freq,
        run,
        args.run_name,
        checkpoint_path=args.checkpoint_path,
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
        direct_horizon_loss_weight=args.direct_horizon_loss_weight,
        rollout_loss_weight=args.rollout_loss_weight,
        forecast_train_horizon=args.forecast_train_horizon,
        rollout_batch_size=args.rollout_batch_size,
    )
    trainer.train(args.epochs, start_epoch=start_epoch)
    print("Training time:", time.time() - start)
    if run is not None:
        run.finish()
    destroy_process_group()


if __name__ == "__main__":
    args = get_args_with_yaml()
    args.run_name = get_run_name(args)
    world_size = torch.cuda.device_count()
    if world_size == 1:
        main(0, world_size, args)
    else:
        mp.spawn(main, args=(world_size, args), nprocs=world_size)
