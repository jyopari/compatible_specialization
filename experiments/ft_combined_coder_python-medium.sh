#!/bin/bash

#SBATCH -p vision-pulkitag-h100
#SBATCH -q vision-pulkitag-main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=gpt_finetune
#SBATCH --output=/data/pulkitag/models/jyop/code/gpt-merge/experiments/out/%x_%j.out   # Output file
#SBATCH --error=/data/pulkitag/models/jyop/code/gpt-merge/experiments/err/%x_%j.err    # Error file

export HOME="/data/pulkitag/models/jyop"
source ~/.bash_profile

cd /data/pulkitag/models/jyop/code/gpt-merge
mamba init
mamba activate gpu_env

python main.py --arch gpt2-medium --pretrained \
            --dataset combined_coder_python --batch_size 64 --grad_accum 4 --context 1024 --train_tokens 160000 \
            --optim adamw --lr 8e-4 --wd 1e-1 --train_mlp\
            --out_dir /data/scratch-oc40/pulkitag/jyop/gpt-merge-out \
            --log_freq 100 --eval_iter 5 \
            --name gpt2-combined_coder_python --wandb 