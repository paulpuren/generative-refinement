# Generative Refinement Learning for Continuous Temporal Interpolation

This repository implements generative refinement learning approaches for high-quality temporal interpolation of spatio-temporal data, with a focus on fluid dynamics simulations and weather data. The project provides multiple model architectures and evaluation frameworks for interpolating between temporal frames in scientific datasets.

## Features

### Models
- **FLEX**: Flexible interpolation model with diffusion-based refinement
- **UNet**: Convolutional neural network for spatial feature extraction
- **SuperSloMo**: Optical flow-based frame interpolation
- **EDEN**: Generative diffusion-based interpolation using denoising processes

### Datasets
- **NSKT**: Navier-Stokes Kolmogorov Turbulence data at various Reynolds numbers (600-36000)
- **Shanghai**: Urban-scale real-world radar dataset
- **SST**: Sea Surface Temperature data

### Key Capabilities
- Multi-scale temporal interpolation (T=8, T=16, T=20 frames)
- Distributed training with PyTorch DDP
- Comprehensive evaluation metrics (RFNE, R² scores)
- WandB integration for experiment tracking
- Patch-based training for large-scale data

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.12+
- CUDA-compatible GPU (recommended for training)

### Setup
1. Copy this folder

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Optional Dependencies
For evaluation and analysis:
```bash
pip install scipy seaborn tqdm lpips
```

## Usage

### Data Preparation
Place your datasets in the `datasets/` directory. The code expects HDF5 files for NSKT data with the following structure:
- Vorticity field (`w`)
- Velocity components (`u`, `v`)

### Training

Use the bash launchers in `scripts/` for training. These wrappers load the expected environment, switch to the repository root, and call the correct Python entry point with the matching config.

#### FLEX
```bash
bash scripts/nskt/train_flex.sh
bash scripts/shanghai/train_flex.sh
bash scripts/sst/train_flex_sst.sh
```

#### UNet
```bash
bash scripts/nskt/train_unet.sh
bash scripts/shanghai/train_unet.sh
bash scripts/sst/train_unet_sst.sh
```

#### SuperSloMo
```bash
bash scripts/nskt/train_super_slomo.sh
bash scripts/shanghai/train_super_slomo.sh
bash scripts/sst/train_super_slomo.sh
```

#### EDEN Baseline
```bash
bash scripts/nskt/train_eden_vae.sh
bash scripts/nskt/train_eden_dit.sh
bash scripts/shanghai/train_eden_vae.sh
bash scripts/shanghai/train_eden_dit.sh
bash scripts/sst/train_eden_vae.sh
bash scripts/sst/train_eden_dit.sh
```

#### Forecasting and Ablations
```bash
bash scripts/forecasting/train_flex_forecast_nskt.sh
bash scripts/forecasting/train_flex_forecast_nskt_posttrain.sh
bash scripts/ablation/train_flex_residual_nskt.sh
bash scripts/ablation/train_flex_no_re_nskt.sh
bash scripts/ablation/train_flex_no_total_interp_steps_nskt.sh
```

Most training scripts accept `CONFIG_PATH` to override the default YAML:
```bash
CONFIG_PATH=config/nskt/flex.yaml bash scripts/nskt/train_flex.sh
```

### Evaluation

Use the evaluation launchers in `scripts/` instead of calling `eval.py` directly.

#### FLEX
```bash
bash scripts/nskt/eval_flex_nskt.sh
bash scripts/shanghai/eval_flex.sh
bash scripts/sst/eval_flex_sst.sh
```

#### UNet
```bash
bash scripts/nskt/eval_unet_nskt.sh
bash scripts/shanghai/eval_unet.sh
bash scripts/sst/eval_unet_sst.sh
```

#### SuperSloMo
```bash
bash scripts/nskt/eval_super_slomo_nskt.sh
bash scripts/shanghai/eval_super_slomo.sh
bash scripts/sst/eval_super_slomo.sh
```

#### EDEN Baseline
```bash
bash scripts/nskt/eval_eden.sh
bash scripts/shanghai/eval_eden.sh
bash scripts/sst/eval_eden.sh
```

#### Forecasting and Ablations
```bash
bash scripts/forecasting/eval_flex_forecast_nskt.sh
bash scripts/ablation/eval_flex_residual_nskt.sh
bash scripts/ablation/eval_flex_no_re_nskt.sh
bash scripts/ablation/eval_flex_no_total_interp_steps_nskt.sh
```

The NSKT FLEX evaluation wrapper supports common overrides:
```bash
RE_ID=5 TOTAL_INTERP_STEPS=20 BATCH_SIZE=12 bash scripts/nskt/eval_flex_nskt.sh
CHECKPOINT_PATH=checkpoints/your_checkpoint_best.pt bash scripts/nskt/eval_flex_nskt.sh
```

### Configuration
Configuration files are located in `config/` directory:
- `nskt/`: NSKT dataset configurations
- `shanghai/`: Shanghai dataset configurations
- `sst/`: SST dataset configurations

Key parameters:
- `model`: Model architecture (flex_small, flex_medium, unet, super_slomo)
- `T`: Number of interpolation frames
- `stride`: Patch stride for training
- `lr`: Learning rate
- `epochs`: Training epochs

## Results

Evaluation results are stored in `assets/` directory with detailed metrics:

### Performance Metrics
- **RFNE (Relative Frobenius Norm Error)**: Measures interpolation accuracy
- **R² Score**: Coefficient of determination for prediction quality
- **SSIM**: Structural similarity evaluations
- **Runtime**: Inference time per sample

## Project Structure

```
Interp-DM/
├── train.py                         # Main FLEX training entry point
├── train_super_slomo.py             # SuperSloMo training entry point
├── train_unet.py                    # UNet training entry point
├── eval.py                          # Main evaluation entry point
├── eval_all.py                      # Batch evaluation utilities (not used)
├── eval_shanghai.py                 # Shanghai evaluation
├── eval_shanghai_update.py          # Updated Shanghai evaluation
├── eval_sst.py                      # SST evaluation
├── eval_super_slomo.py              # SuperSloMo evaluation
├── requirements.txt                 # Python dependencies
├── src/                             # Core model and training utilities
│   ├── flex.py                      # FLEX model implementation
│   ├── unet.py                      # UNet architecture
│   ├── diffusion_model.py           # Diffusion model
│   ├── super_slomo.py               # SuperSloMo implementation
│   ├── metrics.py                   # Evaluation metrics
│   ├── common.py                    # Shared model components
│   ├── helper.py                    # Helper functions
│   ├── lion.py                      # Lion optimizer
│   ├── plotting.py                  # Plotting utilities
│   ├── utilities.py                 # Shared utilities
│   ├── ablation/                    # Ablation diffusion model variants
│   ├── clip/                        # Vendored CLIP dependency
│   └── taming-transformers/         # Vendored taming-transformers dependency
├── datasets/                        # Dataset loaders and preprocessing
│   ├── data_nskt.py                 # NSKT data loader
│   ├── data_nskt_updated.py         # Updated NSKT data loader
│   ├── data_shanghai.py             # Shanghai data loader
│   ├── data_shanghai_updated.py     # Updated Shanghai data loader
│   ├── data_sea_temp.py             # SST data loader
│   ├── data_era5_z500.py            # ERA5 Z500 data loader
│   ├── get_data.py                  # Dataset factory
│   └── preprocess.py                # Dataset preprocessing
├── forecasting/                     # One-step NSKT forecasting code
│   ├── train.py                     # Forecasting training entry point
│   ├── eval.py                      # Forecasting evaluation
│   ├── diffusion_model.py           # Forecasting diffusion model
│   ├── data_nskt.py                 # Forecasting NSKT loader
│   ├── params.py                    # Forecasting parameters
│   └── utilities.py                 # Forecasting utilities
├── config/                          # YAML experiment configurations
│   ├── ablation/                    # Ablation configs
│   ├── forecasting/                 # Forecasting configs
│   ├── nskt/                        # NSKT configs
│   ├── shanghai/                    # Shanghai configs
│   └── sst/                         # SST configs
├── scripts/                         # Shell scripts and benchmarks
│   ├── ablation/                    # Ablation train/eval scripts
│   ├── forecasting/                 # Forecasting train/eval scripts
│   ├── nskt/                        # NSKT train/eval scripts
│   ├── shanghai/                    # Shanghai train/eval scripts
│   ├── sst/                         # SST train/eval scripts
│   └── benchmark_inference_time.py  # Inference benchmark driver
├── EDEN/                            # EDEN baseline implementation
│   ├── train_vae.py                 # EDEN VAE training
│   ├── train_dit.py                 # EDEN DiT training
│   ├── eval.py                      # EDEN evaluation
│   ├── eval_scientific.py           # Scientific dataset evaluation
│   ├── eval_sst.py                  # SST evaluation
│   ├── inference.py                 # EDEN inference
│   ├── configs/                     # EDEN configs
│   ├── src/                         # EDEN source code
│   └── flolpips/                    # FLO-LPIPS metric code
├── analysis/                        # Analysis notebooks, plots, and figure scripts
├── assets/                          # Evaluation results and selected figures
├── checkpoints/                     # Model checkpoint artifacts
├── samples/                         # Generated training samples
├── utils/                           # Parameter and data utility helpers
└── wandb/                           # Weights & Biases run logs
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built on PyTorch and Diffusers library
- Inspired by video frame interpolation methods like SuperSloMo and EDEN
- Uses NSKT turbulence data for benchmarking
