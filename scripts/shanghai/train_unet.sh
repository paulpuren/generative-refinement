# salloc -N 1 -C gpu -q interactive -t 04:00:00 -G 4 -A m5262

module load python
conda activate /pscratch/sd/p/puren93/conda_env/genai
cd ../../

PRETRAINED_MODEL_PATH="checkpoints/checkpoint_Model_UNet_Data_shanghai_Optim_lion_lr1e-05_epoch100_stride64_T20_TfixedFalse.pt"

# run shanghai radar data
python train_unet.py \
    --model UNet \
    --run_name UNet \
    --data_name 'shanghai' \
    --sampling_freq 10 \
    --optimizer 'adam' \
    --epochs 400 \
    --batch_size 12 \
    --learning_rate 1e-4 \
    --checkpoint_path '' \
    --total_interp_steps_train 8 \
    --is_T_fixed False \
    --patch_size 128 \
    --stride 32 \
    --scratch_dir '/global/cfs/cdirs/m4633/puren/interp_dm/shanghai/shanghai.h5'
