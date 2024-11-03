#!/bin/bash

#SBATCH -p vision-pulkitag-h100
#SBATCH -q vision-pulkitag-main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=gpt_finetune
#SBATCH --output=/data/pulkitag/models/jyop/code/gpt-merge/experiments/out/%x_%j.out   # Output file
#SBATCH --error=/data/pulkitag/models/jyop/code/gpt-merge/experiments/err/%x_%j.err    # Error file

export HOME="/data/pulkitag/models/jyop"
source ~/.bash_profile

cd /data/pulkitag/models/jyop/code/gpt-merge
mamba init
mamba activate gpu_env

CUDA_VISIBLE_DEVICES=0 python main.py --arch gpt2 --pretrained \
            --datasets evol_instruct_code --batch_size 92 --context 1024 --train_tokens 160000 --grad_accum 4\
            --optim adamw --lr 5e-3 --wd 1e-1 --router_method standard --topk 6 --multi_layer_router --num_multi_layer_experts 3\
            --router_ckpts /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/gpt2-code_instructions-gpt2-0.0008-1-code_instructions_finetuned.pt \
                           /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/gpt2-tinycodes-gpt2-0.0008-1-tinycodes_finetuned.pt \
            --name gpt_routed_multi_layer_3 --wandb 