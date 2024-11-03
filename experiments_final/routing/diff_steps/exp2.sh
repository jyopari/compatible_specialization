#!/bin/bash

#SBATCH -p vision-pulkitag-h100
#SBATCH -q vision-pulkitag-main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=gpt_route
#SBATCH --output=/data/pulkitag/models/jyop/code/gpt-merge/experiments/out/%x_%j.out   # Output file
#SBATCH --error=/data/pulkitag/models/jyop/code/gpt-merge/experiments/err/%x_%j.err    # Error file

export HOME="/data/pulkitag/models/jyop"
source ~/.bash_profile

cd /data/pulkitag/models/jyop/code/gpt-merge
mamba init
mamba activate gpu_env

CUDA_VISIBLE_DEVICES=0 python main.py --arch gpt2 --pretrained \
            --datasets gsm-hard --batch_size 92 --context 1024 --train_tokens 160000 \
            --optim adamw --lr 5e-3 --wd 1e-1 --router_method standard --topk 2\
            --router_ckpts /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/gpt2-tinycodes-gpt2-0.0008-1-tinycodes_560000_step-8000_finetuned.pt \
                           /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/gpt2-orca-math-gpt2-0.0008-1-orca-math-problems_560000_step-8000_finetuned.pt \
            --name gpt_routed_gsm-hard-8k --wandb 