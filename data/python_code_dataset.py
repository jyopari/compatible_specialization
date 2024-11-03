import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset # huggingface datasets

data_dir = os.path.dirname(__file__)
if not data_dir.endswith('/data'):
    data_dir = os.path.join(data_dir, 'data')
data_dir = os.path.join(data_dir, 'python_code_dataset')

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

num_proc = 8
num_proc_load_dataset = num_proc

if __name__ == '__main__':
    # Load the dataset
    dataset = load_dataset("jtatman/python-code-dataset-500k", num_proc=num_proc_load_dataset)
    
    # Combine 'prompt' and 'response' columns
    def combine_columns(example):
        combined_text = example['instruction'] + " " + example['output']
        return {'combined_text': combined_text}
    
    dataset = dataset.map(combine_columns, num_proc=num_proc, desc="Combining prompt and response")
    
    # Select the combined column
    dataset = dataset.select_columns('combined_text')
    dataset = dataset.rename_column('combined_text', 'text')
    
    # Split the dataset
    split_dataset = dataset["train"].train_test_split(test_size=0.005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test') # Rename the validation split to val

    # Define the encoding function (gpt2 bpe)
    enc = tiktoken.get_encoding("gpt2")
    def process(example):
        ids = enc.encode_ordinary(example['text']) # encode_ordinary ignores any special tokens
        ids.append(enc.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
        out = {'ids': ids, 'len': len(ids)}
        return out

    # Tokenize the dataset
    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )
    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(data_dir, f'{split}.bin')
        dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
