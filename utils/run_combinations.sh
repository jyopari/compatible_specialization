#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python analyze.py --arch gpt2 --pretrained \
            --datasets tinystories --batch_size 16 --context 1024 \
            --name eval_gpt2_base \
            --eval_sep code_instructions code_textbooks coding_dataset combined_coder_python evol_instruct_code programming_books python_code_dataset python_codes tinycodes \
            --router_ckpts /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/gpt2-code_instructions-gpt2-0.0008-1-code_instructions_finetuned.pt \
                           /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/gpt2-combined_coder_python-gpt2-0.0008-1-combined_coder_python_finetuned.pt \
            --analyze

CUDA_VISIBLE_DEVICES=1 python analyze.py --arch gpt2 --pretrained \
            --datasets tinystories --batch_size 16 --context 1024 \
            --name eval_gpt2_base \
            --eval_sep code_instructions code_textbooks coding_dataset combined_coder_python evol_instruct_code programming_books python_code_dataset python_codes tinycodes \
            --router_ckpts /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/gpt2-code_instructions-gpt2-0.0008-1-code_instructions_finetuned.pt \
                           /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/gpt2-python_code_dataset-gpt2-0.0008-1-python_code_dataset_finetuned.pt \
            --analyze

CUDA_VISIBLE_DEVICES=1 python analyze.py --arch gpt2 --pretrained \
            --datasets tinystories --batch_size 16 --context 1024 \
            --name eval_gpt2_base \
            --eval_sep code_instructions code_textbooks coding_dataset combined_coder_python evol_instruct_code programming_books python_code_dataset python_codes tinycodes \
            --router_ckpts /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/gpt2-code_instructions-gpt2-0.0008-1-code_instructions_finetuned.pt \
                           /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/gpt2-tinycodes-gpt2-0.0008-1-tinycodes_finetuned.pt \
            --analyze

CUDA_VISIBLE_DEVICES=1 python analyze.py --arch gpt2 --pretrained \
            --datasets tinystories --batch_size 16 --context 1024 \
            --name eval_gpt2_base \
            --eval_sep code_instructions code_textbooks coding_dataset combined_coder_python evol_instruct_code programming_books python_code_dataset python_codes tinycodes \
            --router_ckpts /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/gpt2-code_instructions-gpt2-0.0008-1-code_instructions_finetuned.pt \
                           /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/v1-gpt2-0.0008-1-programming_books_finetuned.pt \
            --analyze

CUDA_VISIBLE_DEVICES=1 python analyze.py --arch gpt2 --pretrained \
            --datasets tinystories --batch_size 16 --context 1024 \
            --name eval_gpt2_base \
            --eval_sep code_instructions code_textbooks coding_dataset combined_coder_python evol_instruct_code programming_books python_code_dataset python_codes tinycodes \
            --router_ckpts /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/gpt2-code_instructions-gpt2-0.0008-1-code_instructions_finetuned.pt \
                           /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/gpt2-coding_dataset-gpt2-0.0008-1-coding_dataset_finetuned.pt \
            --analyze

CUDA_VISIBLE_DEVICES=1 python analyze.py --arch gpt2 --pretrained \
            --datasets tinystories --batch_size 16 --context 1024 \
            --name eval_gpt2_base \
            --eval_sep code_instructions code_textbooks coding_dataset combined_coder_python evol_instruct_code programming_books python_code_dataset python_codes tinycodes \
            --router_ckpts /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/gpt2-code_instructions-gpt2-0.0008-1-code_instructions_finetuned.pt \
                           /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/gpt2-evol_instruct_code-gpt2-0.0008-1-evol_instruct_code_finetuned.pt \
            --analyze

CUDA_VISIBLE_DEVICES=1 python analyze.py --arch gpt2 --pretrained \
            --datasets tinystories --batch_size 16 --context 1024 \
            --name eval_gpt2_base \
            --eval_sep code_instructions code_textbooks coding_dataset combined_coder_python evol_instruct_code programming_books python_code_dataset python_codes tinycodes \
            --router_ckpts /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/gpt2-code_instructions-gpt2-0.0008-1-code_instructions_finetuned.pt \
                           /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/gpt2-python_codes-gpt2-0.0008-1-python_codes_finetuned.pt \
            --analyze

CUDA_VISIBLE_DEVICES=1 python analyze.py --arch gpt2 --pretrained \
            --datasets tinystories --batch_size 16 --context 1024 \
            --name eval_gpt2_base \
            --eval_sep code_instructions code_textbooks coding_dataset combined_coder_python evol_instruct_code programming_books python_code_dataset python_codes tinycodes \
            --router_ckpts /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/gpt2-code_instructions-gpt2-0.0008-1-code_instructions_finetuned.pt \
                           /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/v1-gpt2-0.0008-1-code_textbooks_finetuned.pt \
            --analyze

CUDA_VISIBLE_DEVICES=1 python analyze.py --arch gpt2 --pretrained \
            --datasets tinystories --batch_size 16 --context 1024 \
            --name eval_gpt2_base \
            --eval_sep code_instructions code_textbooks coding_dataset combined_coder_python evol_instruct_code programming_books python_code_dataset python_codes tinycodes \
            --router_ckpts /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/gpt2-combined_coder_python-gpt2-0.0008-1-combined_coder_python_finetuned.pt \
                           /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/gpt2-python_code_dataset-gpt2-0.0008-1-python_code_dataset_finetuned.pt \
            --analyze

CUDA_VISIBLE_DEVICES=1 python analyze.py --arch gpt2 --pretrained \
            --datasets tinystories --batch_size 16 --context 1024 \
            --name eval_gpt2_base \
            --eval_sep code_instructions code_textbooks coding_dataset combined_coder_python evol_instruct_code programming_books python_code_dataset python_codes tinycodes \
            --router_ckpts /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/gpt2-combined_coder_python-gpt2-0.0008-1-combined_coder_python_finetuned.pt \
                           /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/gpt2-tinycodes-gpt2-0.0008-1-tinycodes_finetuned.pt \
            --analyze

CUDA_VISIBLE_DEVICES=1 python analyze.py --arch gpt2 --pretrained \
            --datasets tinystories --batch_size 16 --context 1024 \
            --name eval_gpt2_base \
            --eval_sep code_instructions code_textbooks coding_dataset combined_coder_python evol_instruct_code programming_books python_code_dataset python_codes tinycodes \
            --router_ckpts /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/gpt2-combined_coder_python-gpt2-0.0008-1-combined_coder_python_finetuned.pt \
                           /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/v1-gpt2-0.0008-1-programming_books_finetuned.pt \
            --analyze

CUDA_VISIBLE_DEVICES=1 python analyze.py --arch gpt2 --pretrained \
            --datasets tinystories --batch_size 16 --context 1024 \
            --name eval_gpt2_base \
            --eval_sep code_instructions code_textbooks coding_dataset combined_coder_python evol_instruct_code programming_books python_code_dataset python_codes tinycodes \
            --router_ckpts /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/gpt2-combined_coder_python-gpt2-0.0008-1-combined_coder_python_finetuned.pt \
                           /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/gpt2-coding_dataset-gpt2-0.0008-1-coding_dataset_finetuned.pt \
            --analyze

CUDA_VISIBLE_DEVICES=1 python analyze.py --arch gpt2 --pretrained \
            --datasets tinystories --batch_size 16 --context 1024 \
            --name eval_gpt2_base \
            --eval_sep code_instructions code_textbooks coding_dataset combined_coder_python evol_instruct_code programming_books python_code_dataset python_codes tinycodes \
            --router_ckpts /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/gpt2-combined_coder_python-gpt2-0.0008-1-combined_coder_python_finetuned.pt \
                           /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/gpt2-evol_instruct_code-gpt2-0.0008-1-evol_instruct_code_finetuned.pt \
            --analyze

CUDA_VISIBLE_DEVICES=1 python analyze.py --arch gpt2 --pretrained \
            --datasets tinystories --batch_size 16 --context 1024 \
            --name eval_gpt2_base \
            --eval_sep code_instructions code_textbooks coding_dataset combined_coder_python evol_instruct_code programming_books python_code_dataset python_codes tinycodes \
            --router_ckpts /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/gpt2-combined_coder_python-gpt2-0.0008-1-combined_coder_python_finetuned.pt \
                           /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/gpt2-python_codes-gpt2-0.0008-1-python_codes_finetuned.pt \
            --analyze

CUDA_VISIBLE_DEVICES=1 python analyze.py --arch gpt2 --pretrained \
            --datasets tinystories --batch_size 16 --context 1024 \
            --name eval_gpt2_base \
            --eval_sep code_instructions code_textbooks coding_dataset combined_coder_python evol_instruct_code programming_books python_code_dataset python_codes tinycodes \
            --router_ckpts /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/gpt2-combined_coder_python-gpt2-0.0008-1-combined_coder_python_finetuned.pt \
                           /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/v1-gpt2-0.0008-1-code_textbooks_finetuned.pt \
            --analyze

CUDA_VISIBLE_DEVICES=1 python analyze.py --arch gpt2 --pretrained \
            --datasets tinystories --batch_size 16 --context 1024 \
            --name eval_gpt2_base \
            --eval_sep code_instructions code_textbooks coding_dataset combined_coder_python evol_instruct_code programming_books python_code_dataset python_codes tinycodes \
            --router_ckpts /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/gpt2-python_code_dataset-gpt2-0.0008-1-python_code_dataset_finetuned.pt \
                           /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/gpt2-tinycodes-gpt2-0.0008-1-tinycodes_finetuned.pt \
            --analyze

CUDA_VISIBLE_DEVICES=1 python analyze.py --arch gpt2 --pretrained \
            --datasets tinystories --batch_size 16 --context 1024 \
            --name eval_gpt2_base \
            --eval_sep code_instructions code_textbooks coding_dataset combined_coder_python evol_instruct_code programming_books python_code_dataset python_codes tinycodes \
            --router_ckpts /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/gpt2-python_code_dataset-gpt2-0.0008-1-python_code_dataset_finetuned.pt \
                           /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/v1-gpt2-0.0008-1-programming_books_finetuned.pt \
            --analyze

CUDA_VISIBLE_DEVICES=1 python analyze.py --arch gpt2 --pretrained \
            --datasets tinystories --batch_size 16 --context 1024 \
            --name eval_gpt2_base \
            --eval_sep code_instructions code_textbooks coding_dataset combined_coder_python evol_instruct_code programming_books python_code_dataset python_codes tinycodes \
            --router_ckpts /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/gpt2-python_code_dataset-gpt2-0.0008-1-python_code_dataset_finetuned.pt \
                           /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/gpt2-coding_dataset-gpt2-0.0008-1-coding_dataset_finetuned.pt \
            --analyze

CUDA_VISIBLE_DEVICES=1 python analyze.py --arch gpt2 --pretrained \
            --datasets tinystories --batch_size 16 --context 1024 \
            --name eval_gpt2_base \
            --eval_sep code_instructions code_textbooks coding_dataset combined_coder_python evol_instruct_code programming_books python_code_dataset python_codes tinycodes \
            --router_ckpts /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/gpt2-python_code_dataset-gpt2-0.0008-1-python_code_dataset_finetuned.pt \
                           /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/gpt2-evol_instruct_code-gpt2-0.0008-1-evol_instruct_code_finetuned.pt \
            --analyze

CUDA_VISIBLE_DEVICES=1 python analyze.py --arch gpt2 --pretrained \
            --datasets tinystories --batch_size 16 --context 1024 \
            --name eval_gpt2_base \
            --eval_sep code_instructions code_textbooks coding_dataset combined_coder_python evol_instruct_code programming_books python_code_dataset python_codes tinycodes \
            --router_ckpts /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/gpt2-python_code_dataset-gpt2-0.0008-1-python_code_dataset_finetuned.pt \
                           /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/gpt2-python_codes-gpt2-0.0008-1-python_codes_finetuned.pt \
            --analyze

CUDA_VISIBLE_DEVICES=1 python analyze.py --arch gpt2 --pretrained \
            --datasets tinystories --batch_size 16 --context 1024 \
            --name eval_gpt2_base \
            --eval_sep code_instructions code_textbooks coding_dataset combined_coder_python evol_instruct_code programming_books python_code_dataset python_codes tinycodes \
            --router_ckpts /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/gpt2-python_code_dataset-gpt2-0.0008-1-python_code_dataset_finetuned.pt \
                           /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/v1-gpt2-0.0008-1-code_textbooks_finetuned.pt \
            --analyze

CUDA_VISIBLE_DEVICES=1 python analyze.py --arch gpt2 --pretrained \
            --datasets tinystories --batch_size 16 --context 1024 \
            --name eval_gpt2_base \
            --eval_sep code_instructions code_textbooks coding_dataset combined_coder_python evol_instruct_code programming_books python_code_dataset python_codes tinycodes \
            --router_ckpts /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/gpt2-tinycodes-gpt2-0.0008-1-tinycodes_finetuned.pt \
                           /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/v1-gpt2-0.0008-1-programming_books_finetuned.pt \
            --analyze

CUDA_VISIBLE_DEVICES=1 python analyze.py --arch gpt2 --pretrained \
            --datasets tinystories --batch_size 16 --context 1024 \
            --name eval_gpt2_base \
            --eval_sep code_instructions code_textbooks coding_dataset combined_coder_python evol_instruct_code programming_books python_code_dataset python_codes tinycodes \
            --router_ckpts /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/gpt2-tinycodes-gpt2-0.0008-1-tinycodes_finetuned.pt \
                           /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/gpt2-coding_dataset-gpt2-0.0008-1-coding_dataset_finetuned.pt \
            --analyze

CUDA_VISIBLE_DEVICES=1 python analyze.py --arch gpt2 --pretrained \
            --datasets tinystories --batch_size 16 --context 1024 \
            --name eval_gpt2_base \
            --eval_sep code_instructions code_textbooks coding_dataset combined_coder_python evol_instruct_code programming_books python_code_dataset python_codes tinycodes \
            --router_ckpts /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/gpt2-tinycodes-gpt2-0.0008-1-tinycodes_finetuned.pt \
                           /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/gpt2-evol_instruct_code-gpt2-0.0008-1-evol_instruct_code_finetuned.pt \
            --analyze

CUDA_VISIBLE_DEVICES=1 python analyze.py --arch gpt2 --pretrained \
            --datasets tinystories --batch_size 16 --context 1024 \
            --name eval_gpt2_base \
            --eval_sep code_instructions code_textbooks coding_dataset combined_coder_python evol_instruct_code programming_books python_code_dataset python_codes tinycodes \
            --router_ckpts /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/gpt2-tinycodes-gpt2-0.0008-1-tinycodes_finetuned.pt \
                           /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/gpt2-python_codes-gpt2-0.0008-1-python_codes_finetuned.pt \
            --analyze

CUDA_VISIBLE_DEVICES=1 python analyze.py --arch gpt2 --pretrained \
            --datasets tinystories --batch_size 16 --context 1024 \
            --name eval_gpt2_base \
            --eval_sep code_instructions code_textbooks coding_dataset combined_coder_python evol_instruct_code programming_books python_code_dataset python_codes tinycodes \
            --router_ckpts /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/gpt2-tinycodes-gpt2-0.0008-1-tinycodes_finetuned.pt \
                           /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/v1-gpt2-0.0008-1-code_textbooks_finetuned.pt \
            --analyze

CUDA_VISIBLE_DEVICES=1 python analyze.py --arch gpt2 --pretrained \
            --datasets tinystories --batch_size 16 --context 1024 \
            --name eval_gpt2_base \
            --eval_sep code_instructions code_textbooks coding_dataset combined_coder_python evol_instruct_code programming_books python_code_dataset python_codes tinycodes \
            --router_ckpts /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/v1-gpt2-0.0008-1-programming_books_finetuned.pt \
                           /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/gpt2-coding_dataset-gpt2-0.0008-1-coding_dataset_finetuned.pt \
            --analyze

CUDA_VISIBLE_DEVICES=1 python analyze.py --arch gpt2 --pretrained \
            --datasets tinystories --batch_size 16 --context 1024 \
            --name eval_gpt2_base \
            --eval_sep code_instructions code_textbooks coding_dataset combined_coder_python evol_instruct_code programming_books python_code_dataset python_codes tinycodes \
            --router_ckpts /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/v1-gpt2-0.0008-1-programming_books_finetuned.pt \
                           /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/gpt2-evol_instruct_code-gpt2-0.0008-1-evol_instruct_code_finetuned.pt \
            --analyze

CUDA_VISIBLE_DEVICES=1 python analyze.py --arch gpt2 --pretrained \
            --datasets tinystories --batch_size 16 --context 1024 \
            --name eval_gpt2_base \
            --eval_sep code_instructions code_textbooks coding_dataset combined_coder_python evol_instruct_code programming_books python_code_dataset python_codes tinycodes \
            --router_ckpts /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/v1-gpt2-0.0008-1-programming_books_finetuned.pt \
                           /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/gpt2-python_codes-gpt2-0.0008-1-python_codes_finetuned.pt \
            --analyze

CUDA_VISIBLE_DEVICES=1 python analyze.py --arch gpt2 --pretrained \
            --datasets tinystories --batch_size 16 --context 1024 \
            --name eval_gpt2_base \
            --eval_sep code_instructions code_textbooks coding_dataset combined_coder_python evol_instruct_code programming_books python_code_dataset python_codes tinycodes \
            --router_ckpts /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/v1-gpt2-0.0008-1-programming_books_finetuned.pt \
                           /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/v1-gpt2-0.0008-1-code_textbooks_finetuned.pt \
            --analyze

CUDA_VISIBLE_DEVICES=1 python analyze.py --arch gpt2 --pretrained \
            --datasets tinystories --batch_size 16 --context 1024 \
            --name eval_gpt2_base \
            --eval_sep code_instructions code_textbooks coding_dataset combined_coder_python evol_instruct_code programming_books python_code_dataset python_codes tinycodes \
            --router_ckpts /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/gpt2-coding_dataset-gpt2-0.0008-1-coding_dataset_finetuned.pt \
                           /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/gpt2-evol_instruct_code-gpt2-0.0008-1-evol_instruct_code_finetuned.pt \
            --analyze

CUDA_VISIBLE_DEVICES=1 python analyze.py --arch gpt2 --pretrained \
            --datasets tinystories --batch_size 16 --context 1024 \
            --name eval_gpt2_base \
            --eval_sep code_instructions code_textbooks coding_dataset combined_coder_python evol_instruct_code programming_books python_code_dataset python_codes tinycodes \
            --router_ckpts /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/gpt2-coding_dataset-gpt2-0.0008-1-coding_dataset_finetuned.pt \
                           /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/gpt2-python_codes-gpt2-0.0008-1-python_codes_finetuned.pt \
            --analyze

CUDA_VISIBLE_DEVICES=1 python analyze.py --arch gpt2 --pretrained \
            --datasets tinystories --batch_size 16 --context 1024 \
            --name eval_gpt2_base \
            --eval_sep code_instructions code_textbooks coding_dataset combined_coder_python evol_instruct_code programming_books python_code_dataset python_codes tinycodes \
            --router_ckpts /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/gpt2-coding_dataset-gpt2-0.0008-1-coding_dataset_finetuned.pt \
                           /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/v1-gpt2-0.0008-1-code_textbooks_finetuned.pt \
            --analyze

CUDA_VISIBLE_DEVICES=1 python analyze.py --arch gpt2 --pretrained \
            --datasets tinystories --batch_size 16 --context 1024 \
            --name eval_gpt2_base \
            --eval_sep code_instructions code_textbooks coding_dataset combined_coder_python evol_instruct_code programming_books python_code_dataset python_codes tinycodes \
            --router_ckpts /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/gpt2-evol_instruct_code-gpt2-0.0008-1-evol_instruct_code_finetuned.pt \
                           /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/gpt2-python_codes-gpt2-0.0008-1-python_codes_finetuned.pt \
            --analyze

CUDA_VISIBLE_DEVICES=1 python analyze.py --arch gpt2 --pretrained \
            --datasets tinystories --batch_size 16 --context 1024 \
            --name eval_gpt2_base \
            --eval_sep code_instructions code_textbooks coding_dataset combined_coder_python evol_instruct_code programming_books python_code_dataset python_codes tinycodes \
            --router_ckpts /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/gpt2-evol_instruct_code-gpt2-0.0008-1-evol_instruct_code_finetuned.pt \
                           /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/v1-gpt2-0.0008-1-code_textbooks_finetuned.pt \
            --analyze

CUDA_VISIBLE_DEVICES=1 python analyze.py --arch gpt2 --pretrained \
            --datasets tinystories --batch_size 16 --context 1024 \
            --name eval_gpt2_base \
            --eval_sep code_instructions code_textbooks coding_dataset combined_coder_python evol_instruct_code programming_books python_code_dataset python_codes tinycodes \
            --router_ckpts /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/gpt2-python_codes-gpt2-0.0008-1-python_codes_finetuned.pt \
                           /data/scratch-oc40/pulkitag/jyop/gpt-merge-out/v1-gpt2-0.0008-1-code_textbooks_finetuned.pt \
            --analyze

