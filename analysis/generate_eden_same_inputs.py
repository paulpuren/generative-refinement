from pathlib import Path
import argparse
import sys
from types import SimpleNamespace

import numpy as np
import torch
import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate EDEN NSKT predictions for the exact start/end frames in a comparison cache."
    )
    parser.add_argument("--comparison_cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--vae_checkpoint_path", type=Path, required=True)
    parser.add_argument("--dit_checkpoint_path", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    cli_args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    eden_dir = repo_root / "EDEN"
    if str(eden_dir) not in sys.path:
        sys.path.insert(0, str(eden_dir))

    from eval_scientific import sample_with_eden
    from src.models import load_model
    from src.transport import Sampler, create_transport

    if not torch.cuda.is_available():
        raise RuntimeError("Generating EDEN predictions requires CUDA.")

    with open(cli_args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    args = SimpleNamespace(**config)
    args.vae_checkpoint_path = str(cli_args.vae_checkpoint_path)
    args.dit_checkpoint_path = str(cli_args.dit_checkpoint_path)

    torch.manual_seed(cli_args.seed)
    np.random.seed(cli_args.seed)
    device = torch.device("cuda")

    cache = np.load(cli_args.comparison_cache, allow_pickle=False)
    cond_start = torch.from_numpy(cache["start"]).float().unsqueeze(0).unsqueeze(0).to(device)
    cond_end = torch.from_numpy(cache["end"]).float().unsqueeze(0).unsqueeze(0).to(device)
    total_interp_steps = int(cache["ground_truth"].shape[0])

    dit = load_model("EDEN_DiT", **args.model_args).to(device)
    vae = load_model("EDEN_VAE", **args.vae_args).to(device)

    dit_ckpt = torch.load(args.dit_checkpoint_path, map_location="cpu")
    vae_ckpt = torch.load(args.vae_checkpoint_path, map_location="cpu")
    dit.load_state_dict(dit_ckpt["eden_dit"])
    vae.load_state_dict(vae_ckpt["eden_vae"], strict=False)
    dit.eval()
    vae.eval()

    sampler = Sampler(create_transport(**args.transport))
    sample_fn = sampler.sample_ode(
        sampling_method=args.sampling_method,
        num_steps=args.sample_steps,
        atol=args.atol,
        rtol=args.rtol,
    )

    predictions = []
    with torch.no_grad():
        for step in range(1, total_interp_steps + 1):
            interp = torch.tensor(
                [step / (total_interp_steps + 1.0)],
                dtype=torch.float32,
                device=device,
            )
            pred = sample_with_eden(
                dit=dit,
                vae=vae,
                sample_fn=sample_fn,
                cond_start=cond_start,
                cond_end=cond_end,
                interp=interp,
                args=args,
                device=device,
            )
            predictions.append(pred.detach().cpu().float().squeeze().numpy())

    cli_args.output.parent.mkdir(parents=True, exist_ok=True)
    re_value = (
        np.asarray(cache["re_value"], dtype=np.int64)
        if "re_value" in cache.files
        else np.asarray(-1, dtype=np.int64)
    )
    np.savez_compressed(
        cli_args.output,
        sample_index=np.asarray(cache["sample_index"], dtype=np.int64),
        re_value=re_value,
        eden=np.stack(predictions, axis=0).astype(np.float32),
    )
    print(f"Saved EDEN same-input predictions to {cli_args.output}")


if __name__ == "__main__":
    main()
