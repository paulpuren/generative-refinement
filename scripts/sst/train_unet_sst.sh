# salloc -N 1 -C gpu -q interactive -t 04:00:00 -G 4 -A m4633

module load python
conda activate /pscratch/sd/p/puren93/conda_env/genai
cd ../../

PRETRAINED_PATH="checkpoints/checkpoint_Model_UNet_Data_sea_temp_Optim_adam_lr0.0001_epoch30_stride64_T10_TfixedFalse.pt"

# run sea temperature data
python train_unet.py \
    --run_name UNet \
    --data_name 'sea_temp' \
    --model UNet \
    --sampling_freq 2 \
    --optimizer 'adam' \
    --epochs 30 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --checkpoint_path $PRETRAINED_PATH \
    --total_interp_steps_train 10 \
    --patch_size 128 \
    --stride 64 \
    --is_T_fixed False \
    --scratch_dir '/global/cfs/projectdirs/m4633/puren/interp_dm/sea_temp/'

#     --checkpoint_path '' \