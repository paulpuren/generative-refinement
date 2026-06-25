"""
One-step FLEX forecasting diffusion model.

The model denoises a forecast refinement around a baseline estimate. By
default, the baseline is persistence, so the denoising target is
``x_{t+1} - x_t``. A zero future endpoint is passed through the FLEX
conditioning path so the backbone shape matches interpolation runs without
leaking a future frame.
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


class ForecastingDiffusionModel(nn.Module):
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
        forecast_baseline: str = "persistence",
        forecast_conditioning_mode: str = "forecast",
    ) -> None:
        super().__init__()
        assert prediction_type in {"v", "eps", "x"}
        if forecast_baseline not in {"persistence", "zero"}:
            raise ValueError("forecast_baseline must be one of 'persistence' or 'zero'.")
        if forecast_conditioning_mode not in {"forecast", "interpolation"}:
            raise ValueError(
                "forecast_conditioning_mode must be one of 'forecast' or 'interpolation'."
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
        self.forecast_baseline = forecast_baseline
        self.forecast_conditioning_mode = forecast_conditioning_mode

        self._encoder_forward_params = set(inspect.signature(self.encoder.forward).parameters)
        self._decoder_forward_params = set(inspect.signature(self.decoder.forward).parameters)
        self._task_encoder_forward_params = set(inspect.signature(self.task_encoder.forward).parameters)

    def _estimate_snapshot(self, cond_snapshot_start: torch.Tensor) -> torch.Tensor:
        if self.forecast_baseline == "persistence":
            return cond_snapshot_start
        return torch.zeros_like(cond_snapshot_start)

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
        target_step = self._flatten_scalar(target_interp_step).float()
        total_steps = self._flatten_scalar(total_interp_steps).float()
        if self.forecast_conditioning_mode == "forecast":
            tau = target_step / self.dt_normalization_scale
        else:
            tau = target_step / (total_steps + 1.0)
        if self.condition_on_total_interp_steps:
            dt = total_steps / self.dt_normalization_scale
        else:
            dt = torch.zeros_like(tau)
        if self.condition_on_re:
            reynolds = self._flatten_scalar(fluid_condition).float()
        else:
            reynolds = torch.ones_like(tau)
        return {"tau": tau, "dt": dt, "reynolds": reynolds}

    def _effective_fluid_condition(self, fluid_condition):
        return fluid_condition if self.condition_on_re else None

    def _effective_total_interp_steps(self, total_interp_steps):
        return total_interp_steps if self.condition_on_total_interp_steps else None

    def _run_task_encoder(
        self,
        cond_snapshot_start,
        cond_snapshot_end,
        fluid_condition,
        target_interp_step,
        total_interp_steps,
    ):
        encoder_kwargs = {}
        if "fluid_condition" in self._task_encoder_forward_params:
            encoder_kwargs["fluid_condition"] = fluid_condition
        if "target_interp_step" in self._task_encoder_forward_params:
            encoder_kwargs["target_interp_step"] = target_interp_step
        if "total_interp_steps" in self._task_encoder_forward_params:
            encoder_kwargs["total_interp_steps"] = total_interp_steps
        if getattr(self.task_encoder, "expects_separate_endpoints", False):
            return self.task_encoder(cond_snapshot_start, cond_snapshot_end, **encoder_kwargs)
        return self.task_encoder(torch.cat((cond_snapshot_start, cond_snapshot_end), dim=1), **encoder_kwargs)

    def _run_encoder(
        self,
        residual_t,
        t,
        fluid_condition,
        target_interp_step,
        total_interp_steps,
        scalar_cond,
        spatial_cond_maps,
    ):
        kwargs = {"fluid_condition": fluid_condition, "cond_skips": None}
        if "target_interp_step" in self._encoder_forward_params:
            kwargs["target_interp_step"] = target_interp_step
        if "total_interp_steps" in self._encoder_forward_params:
            kwargs["total_interp_steps"] = total_interp_steps
        if "scalar_cond" in self._encoder_forward_params:
            kwargs["scalar_cond"] = scalar_cond
        if "spatial_cond_maps" in self._encoder_forward_params:
            kwargs["spatial_cond_maps"] = spatial_cond_maps
        return self.encoder(residual_t, t, **kwargs)

    def _run_decoder(
        self,
        h,
        skips,
        t,
        fluid_condition,
        target_interp_step,
        total_interp_steps,
        scalar_cond,
        spatial_cond_maps,
    ):
        kwargs = {"fluid_condition": fluid_condition}
        if "target_interp_step" in self._decoder_forward_params:
            kwargs["target_interp_step"] = target_interp_step
        if "total_interp_steps" in self._decoder_forward_params:
            kwargs["total_interp_steps"] = total_interp_steps
        if "scalar_cond" in self._decoder_forward_params:
            kwargs["scalar_cond"] = scalar_cond
        if "spatial_cond_maps" in self._decoder_forward_params:
            kwargs["spatial_cond_maps"] = spatial_cond_maps
        decoder_out = self.decoder(h, skips, None, None, t, **kwargs)
        if isinstance(decoder_out, tuple):
            return decoder_out
        return decoder_out, {}

    def forward(
        self,
        target_snapshot: torch.Tensor,
        cond_snapshot_start: torch.Tensor,
        cond_snapshot_end: torch.Tensor,
        fluid_condition: torch.Tensor,
        target_interp_step=torch.Tensor,
        total_interp_steps=torch.Tensor,
    ) -> torch.Tensor:
        est_snapshot = self._estimate_snapshot(cond_snapshot_start)
        refinement = target_snapshot - est_snapshot

        t = torch.rand(refinement.shape[0], device=refinement.device)
        _, alpha, sigma = get_logsnr_alpha_sigma(t, shift=self.logsnr_shift)
        eps = torch.randn_like(refinement, device=refinement.device)
        residual_t = alpha * refinement + sigma * eps

        scalar_cond = self._build_scalar_cond(fluid_condition, target_interp_step, total_interp_steps)
        effective_fluid_condition = self._effective_fluid_condition(fluid_condition)
        effective_total_interp_steps = self._effective_total_interp_steps(total_interp_steps)
        spatial_cond_maps = self._run_task_encoder(
            cond_snapshot_start,
            cond_snapshot_end,
            effective_fluid_condition,
            target_interp_step,
            effective_total_interp_steps,
        )
        h, skips = self._run_encoder(
            residual_t,
            t,
            effective_fluid_condition,
            target_interp_step,
            effective_total_interp_steps,
            scalar_cond,
            spatial_cond_maps,
        )
        pred, aux_losses = self._run_decoder(
            h,
            skips,
            t,
            effective_fluid_condition,
            target_interp_step,
            effective_total_interp_steps,
            scalar_cond,
            spatial_cond_maps,
        )

        if self.prediction_type == "x":
            target = refinement
        elif self.prediction_type == "eps":
            pred = alpha * pred + sigma * residual_t
            target = eps
        else:
            target = alpha * eps - sigma * refinement

        loss = self.criterion(pred, target)
        if aux_losses:
            loss = loss + sum(aux_losses.values())
        return loss

    def _sample_impl(
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
        est_snapshot = self._estimate_snapshot(cond_snapshot_start)
        fluid_condition = fluid_condition.to(device)
        target_interp_step = target_interp_step.to(device)
        total_interp_steps = total_interp_steps.to(device)

        scalar_cond = self._build_scalar_cond(fluid_condition, target_interp_step, total_interp_steps)
        effective_fluid_condition = self._effective_fluid_condition(fluid_condition)
        effective_total_interp_steps = self._effective_total_interp_steps(total_interp_steps)
        spatial_cond_maps = self._run_task_encoder(
            cond_snapshot_start,
            cond_snapshot_end,
            effective_fluid_condition,
            target_interp_step,
            effective_total_interp_steps,
        )

        mean = snapshots_i
        for time_step in range(self.diff_steps, 0, -1):
            t = torch.full((n_sample,), time_step / self.diff_steps, device=device)
            t_prev = torch.full((n_sample,), (time_step - 1) / self.diff_steps, device=device)
            _, alpha, sigma = get_logsnr_alpha_sigma(t, shift=self.logsnr_shift)
            _, alpha_prev, sigma_prev = get_logsnr_alpha_sigma(t_prev, shift=self.logsnr_shift)

            h, skips = self._run_encoder(
                snapshots_i,
                t,
                effective_fluid_condition,
                target_interp_step,
                effective_total_interp_steps,
                scalar_cond,
                spatial_cond_maps,
            )
            pred, _ = self._run_decoder(
                h,
                skips,
                t,
                effective_fluid_condition,
                target_interp_step,
                effective_total_interp_steps,
                scalar_cond,
                spatial_cond_maps,
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

            snapshots_i = alpha_prev * mean + sigma_prev * eps

        return mean + est_snapshot

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
        return self._sample_impl(
            n_sample,
            size,
            cond_snapshot_start,
            cond_snapshot_end,
            fluid_condition,
            target_interp_step,
            total_interp_steps,
            device,
            snapshots_i,
        )

    def sample_differentiable(
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
        return self._sample_impl(
            n_sample,
            size,
            cond_snapshot_start,
            cond_snapshot_end,
            fluid_condition,
            target_interp_step,
            total_interp_steps,
            device,
            snapshots_i,
        )
