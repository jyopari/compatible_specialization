import itertools

# List of checkpoints
checkpoints = [
    "gpt2-code_instructions-gpt2-0.0008-1-code_instructions_finetuned.pt",
    "gpt2-combined_coder_python-gpt2-0.0008-1-combined_coder_python_finetuned.pt",
    "gpt2-python_code_dataset-gpt2-0.0008-1-python_code_dataset_finetuned.pt",
    "gpt2-tinycodes-gpt2-0.0008-1-tinycodes_finetuned.pt",
    "v1-gpt2-0.0008-1-programming_books_finetuned.pt",
    "gpt2-coding_dataset-gpt2-0.0008-1-coding_dataset_finetuned.pt",
    "gpt2-evol_instruct_code-gpt2-0.0008-1-evol_instruct_code_finetuned.pt",
    "gpt2-python_codes-gpt2-0.0008-1-python_codes_finetuned.pt",
    "v1-gpt2-0.0008-1-code_textbooks_finetuned.pt"
]

# Directory path
directory = "/data/scratch-oc40/pulkitag/jyop/gpt-merge-out/"

# Create all combinations of pairs
combinations = list(itertools.combinations(checkpoints, 2))

# Create the bash script
with open("run_combinations.sh", "w") as f:
    f.write("#!/bin/bash\n\n")
    for combo in combinations:
        ckpt1 = f"{directory}{combo[0]}"
        ckpt2 = f"{directory}{combo[1]}"
        command = f"CUDA_VISIBLE_DEVICES=1 python analyze.py --arch gpt2 --pretrained \\\n"
        command += f"            --datasets tinystories --batch_size 16 --context 1024 \\\n"
        command += f"            --name eval_gpt2_base \\\n"
        command += f"            --eval_sep code_instructions code_textbooks coding_dataset combined_coder_python evol_instruct_code programming_books python_code_dataset python_codes tinycodes \\\n"
        command += f"            --router_ckpts {ckpt1} \\\n"
        command += f"                           {ckpt2} \\\n"
        command += f"            --analyze\n\n"
        f.write(command)

print("Bash script 'run_combinations.sh' has been created.")