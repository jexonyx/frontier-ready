#!/usr/bin/env python3
"""
FineWeb-Edu dataset preparation (for pretraining).
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu

Downloads and tokenizes the data and saves data shards to disk.

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --output-dir /path/to/output
"""
import os
import argparse
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm


def tokenize(doc, enc, eot):
    """Tokenize a single document and returns a numpy array of uint16 tokens."""
    tokens = [eot]  # the special <|endoftext|> token delimits all documents
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16


def write_datafile(filename, tokens_np):
    """Write tokens to a numpy file."""
    np.save(filename, tokens_np)


def main():
    parser = argparse.ArgumentParser(description='Prepare FineWeb-Edu dataset')
    parser.add_argument('--output-dir', type=str, default='edu_fineweb10B',
                       help='output directory for data shards')
    parser.add_argument('--dataset-name', type=str, default='sample-10BT',
                       help='FineWeb-Edu dataset name (e.g., sample-10BT, sample-100BT)')
    parser.add_argument('--shard-size', type=int, default=int(1e8),
                       help='tokens per shard (default: 100M)')
    parser.add_argument('--num-proc', type=int, default=None,
                       help='number of processes (default: half of CPU count)')
    args = parser.parse_args()

    local_dir = args.output_dir
    remote_name = args.dataset_name
    shard_size = args.shard_size

    # Create the output directory
    if os.path.exists("/data"):
        DATA_CACHE_DIR = os.path.join("/data", local_dir)
    else:
        DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", local_dir)
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    print(f"Output directory: {DATA_CACHE_DIR}")
    print(f"Dataset: HuggingFaceFW/fineweb-edu, name={remote_name}")
    print(f"Shard size: {shard_size:,} tokens")

    # Download the dataset
    print("Loading dataset...")
    fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")

    # Initialize the tokenizer
    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens['<|endoftext|>']  # end of text token

    # Tokenize all documents and write output shards
    # Set multiprocessing start method to 'fork' for compatibility
    mp.set_start_method('fork', force=True)
    nprocs = args.num_proc if args.num_proc else max(1, os.cpu_count()//2)
    print(f"Using {nprocs} processes for tokenization")

    with mp.Pool(nprocs) as pool:
        shard_index = 0
        # Preallocate buffer to hold current shard
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar = None

        # Create a wrapper function that includes enc and eot
        def tokenize_wrapper(doc):
            return tokenize(doc, enc, eot)

        for tokens in pool.imap(tokenize_wrapper, fw, chunksize=16):
            # Is there enough space in the current shard for the new tokens?
            if token_count + len(tokens) < shard_size:
                # Simply append tokens to current shard
                all_tokens_np[token_count:token_count+len(tokens)] = tokens
                token_count += len(tokens)
                # Update progress bar
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                # Write the current shard and start a new one
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
                # Split the document into whatever fits in this shard; the remainder goes to next one
                remainder = shard_size - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
                write_datafile(filename, all_tokens_np)
                shard_index += 1
                progress_bar = None
                # Populate the next shard with the leftovers of the current doc
                all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
                token_count = len(tokens)-remainder

        # Write any remaining tokens as the last shard
        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
            write_datafile(filename, all_tokens_np[:token_count])

    print(f"\nDone! Created {shard_index + 1} shards in {DATA_CACHE_DIR}")


if __name__ == "__main__":
    main()
