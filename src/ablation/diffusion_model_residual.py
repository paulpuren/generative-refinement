"""
A FLEX-style diffusion model with separated denoising and conditioning paths.

This ablation denoises the residual from the first conditioning snapshot to the
target snapshot: target - x0. The second conditioning snapshot is replaced with
zeros so the model cannot condition on the future endpoint, making this behave
like a forecasting setup. At sampling time the generated residual is added back
to x0.
"""

import inspect
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


def logsnr_schedule_cosine(
        t: torch.Tensor,
        logsnr_min: float = -20.0,
        logsnr_max: float = 20.0,
        shift: float = 1.0,
    ) -> torch.Tensor:
    b = torch.atan(torch.exp(-0.5 * torch.tensor(logsnr_max)))
    a = torch.atan(torch.exp(-0.5 * torch.tensor(logsnr_min))) - b
    return -2.0 * torch.log(torch.tan(a * t + b) * shift)


def get_logsnr_alpha_sigma(
        time: torch.Tensor,
        shift: float = 16.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logsnr = logsnr_schedule_cosine(time, shift=shift)[:, None, None, None]
    alpha = torch.sqrt(torch.sigmoid(logsnr))
    sigma = torch.sqrt(torch.sigmoid(-logsnr))
    return logsnr, alpha, sigma


class DiffusionModel(nn.Module):
    """Diffusion model using separated scalar and spatial conditioning."""

    def __init__(
            self,
            encoder: nn.Module,
            decoder: nn.Module,
            task_encoder: nn.Module,
            task_encoder_end: Optional[nn.Module],
            diff_steps: int,
            prediction_type: str,
            criterion: Optional[nn.Module] = None,
            logsnr_shift: float = 1.0,
            dt_normalization_scale: float = 1.0,
            condition_on_re: bool = True,
            condition_on_total_interp_steps: bool = True,
        ) -> None:
        super().__init__()
        assert prediction_type in {"v", "eps", "x"}, (
            "Prediction_type must be one of 'v', 'eps', 'x'"
        )
        self.prediction_type = prediction_type
        self.encoder = encoder
        self.decoder = decoder
        self.task_encoder = task_encoder
        self.task_encoder_end = task_encoder_end
        self.diff_steps = diff_steps
        self.criterion = criterion
        self.logsnr_shift = logsnr_shift
        self.dt_normalization_scale = max(float(dt_normalization_scale), 1.0)
        self.condition_on_re = bool(condition_on_re)
        self.condition_on_total_interp_steps = bool(condition_on_total_interp_steps)

        self._encoder_forward_params = set(inspect.signature(self.encoder.forward).parameters)
        self._decoder_forward_params = set(inspect.signature(self.decoder.forward).parameters)
        self._task_encoder_forward_params = set(inspect.signature(self.task_encoder.forward).parameters)

    @staticmethod
    def _estimate_snapshot(
            cond_snapshot_start: torch.Tensor,
            cond_snapshot_end: torch.Tensor,
            target_interp_step: torch.Tensor,
            total_interp_steps: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        target_interp_step = target_interp_step.float()
        total_interp_steps = total_interp_steps.float()
        delta = target_interp_step / (total_interp_steps + 1.0)
        while delta.ndim < cond_snapshot_start.ndim:
            delta = delta[..., None]
        est_snapshot = (1.0 - delta) * cond_snapshot_start + delta * cond_snapshot_end
        return est_snapshot, delta.reshape(delta.shape[0], -1)[:, 0]

    @staticmethod
    def _flatten_scalar(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 0:
            return x[None]
        if x.ndim == 1:
            return x
        return x.reshape(x.shape[0], -1)[:, 0]

    def _build_scalar_cond(
            self,
            fluid_condition: torch.Tensor,
            target_interp_step: torch.Tensor,
            total_interp_steps: torch.Tensor,
        ) -> Dict[str, torch.Tensor]:
        tau = self._flatten_scalar(target_interp_step).float() / (
            self._flatten_scalar(total_interp_steps).float() + 1.0
        )
        if self.condition_on_total_interp_steps:
            # Normalize dt by a fixed dataset/config scale so it stays comparable to tau.
            dt = self._flatten_scalar(total_interp_steps).float() / self.dt_normalization_scale
        else:
            dt = torch.zeros_like(tau)
        if self.condition_on_re:
            reynolds = self._flatten_scalar(fluid_condition).float()
        else:
            reynolds = torch.ones_like(tau)
        return {
            "tau": tau,
            "dt": dt,
            "reynolds": reynolds,
        }

    def _effective_fluid_condition(
            self,
            fluid_condition: Optional[torch.Tensor],
        ) -> Optional[torch.Tensor]:
        if not self.condition_on_re:
            return None
        return fluid_condition

    def _effective_total_interp_steps(
            self,
            total_interp_steps: Optional[torch.Tensor],
        ) -> Optional[torch.Tensor]:
        if not self.condition_on_total_interp_steps:
            return None
        return total_interp_steps

    def _run_task_encoder(
            self,
            cond_snapshot_start: torch.Tensor,
            cond_snapshot_end: torch.Tensor,
            fluid_condition: Optional[torch.Tensor],
            target_interp_step: torch.Tensor,
            total_interp_steps: torch.Tensor,
        ):
        encoder_kwargs = {}
        if "fluid_condition" in self._task_encoder_forward_params:
            encoder_kwargs["fluid_condition"] = fluid_condition
        if "target_interp_step" in self._task_encoder_forward_params:
            encoder_kwargs["target_interp_step"] = target_interp_step
        if "total_interp_steps" in self._task_encoder_forward_params:
            encoder_kwargs["total_interp_steps"] = total_interp_steps

        if getattr(self.task_encoder, "expects_separate_endpoints", False):
            return self.task_encoder(
                cond_snapshot_start,
                cond_snapshot_end,
                **encoder_kwargs,
            )

        task_input = torch.cat((cond_snapshot_start, cond_snapshot_end), dim=1)
        return self.task_encoder(task_input, **encoder_kwargs)

    @staticmethod
    def _zero_future_endpoint(cond_snapshot_end: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(cond_snapshot_end)

    def _run_encoder(
            self,
            residual_t: torch.Tensor,
            t: torch.Tensor,
            fluid_condition: torch.Tensor,
            target_interp_step: torch.Tensor,
            total_interp_steps: torch.Tensor,
            scalar_cond: Dict[str, torch.Tensor],
            spatial_cond_maps,
        ):
        encoder_kwargs = {
            "fluid_condition": fluid_condition,
            "cond_skips": None,
        }
        if "target_interp_step" in self._encoder_forward_params:
            encoder_kwargs["target_interp_step"] = target_interp_step
        if "total_interp_steps" in self._encoder_forward_params:
            encoder_kwargs["total_interp_steps"] = total_interp_steps
        if "scalar_cond" in self._encoder_forward_params:
            encoder_kwargs["scalar_cond"] = scalar_cond
        if "spatial_cond_maps" in self._encoder_forward_params:
            encoder_kwargs["spatial_cond_maps"] = spatial_cond_maps
        return self.encoder(residual_t, t, **encoder_kwargs)

    def _run_decoder(
            self,
            h: torch.Tensor,
            skips,
            t: torch.Tensor,
            fluid_condition: torch.Tensor,
            target_interp_step: torch.Tensor,
            total_interp_steps: torch.Tensor,
            scalar_cond: Dict[str, torch.Tensor],
            spatial_cond_maps,
        ):
        decoder_kwargs = {
            "fluid_condition": fluid_condition,
        }
        if "target_interp_step" in self._decoder_forward_params:
            decoder_kwargs["target_interp_step"] = target_interp_step
        if "total_interp_steps" in self._decoder_forward_params:
            decoder_kwargs["total_interp_steps"] = total_interp_steps
        if "scalar_cond" in self._decoder_forward_params:
            decoder_kwargs["scalar_cond"] = scalar_cond
        if "spatial_cond_maps" in self._decoder_forward_params:
            decoder_kwargs["spatial_cond_maps"] = spatial_cond_maps

        decoder_out = self.decoder(
            h,
            skips,
            None,
            None,
            t,
            **decoder_kwargs,
        )
        if isinstance(decoder_out, tuple):
            pred, aux_losses = decoder_out
        else:
            pred, aux_losses = decoder_out, {}
        return pred, aux_losses

    def forward(
            self,
            target_snapshot: torch.Tensor,
            cond_snapshot_start: torch.Tensor,
            cond_snapshot_end: torch.Tensor,
            fluid_condition: torch.Tensor,
            target_interp_step=torch.Tensor,
            total_interp_steps=torch.Tensor,
        ) -> torch.Tensor:
        cond_snapshot_end = self._zero_future_endpoint(cond_snapshot_end)
        residual = target_snapshot - cond_snapshot_start

        t = torch.rand(residual.shape[0], device=residual.device)
        _, alpha, sigma = get_logsnr_alpha_sigma(t, shift=self.logsnr_shift)
        eps = torch.randn_like(residual, device=residual.device)
        residual_t = alpha * residual + sigma * eps

        scalar_cond = self._build_scalar_cond(
            fluid_condition=fluid_condition,
            target_interp_step=target_interp_step,
            total_interp_steps=total_interp_steps,
        )
        effective_fluid_condition = self._effective_fluid_condition(fluid_condition)
        effective_total_interp_steps = self._effective_total_interp_steps(total_interp_steps)
        spatial_cond_maps = self._run_task_encoder(
            cond_snapshot_start=cond_snapshot_start,
            cond_snapshot_end=cond_snapshot_end,
            fluid_condition=effective_fluid_condition,
            target_interp_step=target_interp_step,
            total_interp_steps=effective_total_interp_steps,
        )

        h, skips = self._run_encoder(
            residual_t=residual_t,
            t=t,
            fluid_condition=effective_fluid_condition,
            target_interp_step=target_interp_step,
            total_interp_steps=effective_total_interp_steps,
            scalar_cond=scalar_cond,
            spatial_cond_maps=spatial_cond_maps,
        )

        pred, aux_losses = self._run_decoder(
            h=h,
            skips=skips,
            t=t,
            fluid_condition=effective_fluid_condition,
            target_interp_step=target_interp_step,
            total_interp_steps=effective_total_interp_steps,
            scalar_cond=scalar_cond,
            spatial_cond_maps=spatial_cond_maps,
        )

        if self.prediction_type == "x":
            target = residual
        elif self.prediction_type == "eps":
            pred = alpha * pred + sigma * residual_t
            target = eps
        else:
            target = alpha * eps - sigma * residual

        loss = self.criterion(pred, target)
        if aux_losses:
            loss = loss + sum(aux_losses.values())
        return loss

    @torch.no_grad()
    def sample(
            self,
            n_sample: int,
            size: Tuple[int, int, int],
            cond_snapshot_start: torch.Tensor,
            cond_snapshot_end: torch.Tensor,
            fluid_condition: torch.Tensor,
            target_interp_step: torch.Tensor,
            total_interp_steps: torch.Tensor,
            device: str = "cuda",
            snapshots_i: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
        if snapshots_i is None:
            snapshots_i = torch.randn(n_sample, *size, device=device)

        cond_snapshot_start = cond_snapshot_start.to(device)
        cond_snapshot_end = cond_snapshot_end.to(device)
        fluid_condition = fluid_condition.to(device)
        target_interp_step = target_interp_step.to(device)
        total_interp_steps = total_interp_steps.to(device)
        cond_snapshot_end = self._zero_future_endpoint(cond_snapshot_end)

        scalar_cond = self._build_scalar_cond(
            fluid_condition=fluid_condition,
            target_interp_step=target_interp_step,
            total_interp_steps=total_interp_steps,
        )
        effective_fluid_condition = self._effective_fluid_condition(fluid_condition)
        effective_total_interp_steps = self._effective_total_interp_steps(total_interp_steps)
        spatial_cond_maps = self._run_task_encoder(
            cond_snapshot_start=cond_snapshot_start,
            cond_snapshot_end=cond_snapshot_end,
            fluid_condition=effective_fluid_condition,
            target_interp_step=target_interp_step,
            total_interp_steps=effective_total_interp_steps,
        )

        mean = snapshots_i
        for time_step in range(self.diff_steps, 0, -1):
            t = torch.full((n_sample,), time_step / self.diff_steps, device=device)
            t_prev = torch.full((n_sample,), (time_step - 1) / self.diff_steps, device=device)

            _, alpha, sigma = get_logsnr_alpha_sigma(t, shift=self.logsnr_shift)
            _, alpha_prev, sigma_prev = get_logsnr_alpha_sigma(t_prev, shift=self.logsnr_shift)

            h, skips = self._run_encoder(
                residual_t=snapshots_i,
                t=t,
                fluid_condition=effective_fluid_condition,
                target_interp_step=target_interp_step,
                total_interp_steps=effective_total_interp_steps,
                scalar_cond=scalar_cond,
                spatial_cond_maps=spatial_cond_maps,
            )

            pred, _ = self._run_decoder(
                h=h,
                skips=skips,
                t=t,
                fluid_condition=effective_fluid_condition,
                target_interp_step=target_interp_step,
                total_interp_steps=effective_total_interp_steps,
                scalar_cond=scalar_cond,
                spatial_cond_maps=spatial_cond_maps,
            )

            if self.prediction_type == "v":
                mean = alpha * snapshots_i - sigma * pred
                eps = alpha * pred + sigma * snapshots_i
            elif self.prediction_type == "x":
                mean = pred
                eps = (alpha * pred - snapshots_i) / sigma
            else:
                mean = alpha * snapshots_i - sigma * pred
                eps = alpha * pred + sigma * snapshots_i

            eta = 0.0
            noise = torch.randn_like(snapshots_i, device=device) if eta > 0 else 0.0
            snapshots_i = alpha_prev * mean + sigma_prev * eps + eta * sigma_prev * noise

        return mean + cond_snapshot_start
