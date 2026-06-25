# salloc -N 1 -C gpu -q interactive -t 04:00:00 -G 4 -A m4633

module load python
conda activate /pscratch/sd/p/puren93/conda_env/genai
cd ../../

PRETRAINED_CHECKPOINT="./checkpoints/checkpoint_Model_UNet_Data_nskt_Optim_adam_lr0.0001_epoch200_stride32_T20_TfixedFalse.pt"

# select 12000, 24000, and 36000: 5, 7, 9 for eval OOD generalization

# run nskt
python eval.py \
    --model UNet \
    --data_name 'nskt' \
    --re_id 9 \
    --optimizer 'adam' \
    --epochs 200 \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --total_interp_steps 16 \
    --total_interp_steps_train 20 \
    --is_T_fixed False \
    --patch_size 256 \
    --stride 32 \
    --checkpoint_path $PRETRAINED_CHECKPOINT \
    --scratch_dir '/global/cfs/cdirs/m4633/foundationmodel/nskt_tensor/'