import sys
from glob import glob
import random as rnd
import os
import re
import shutil
import numpy as np
from tqdm import tqdm

rnd.seed(42)


def main(input_dir: str, sample_size: int = 3000) -> None:
    fixed_name = "dataset"

    source_path = glob(os.path.join(input_dir, f"*{fixed_name}*.sl"))[0]
    target_path = glob(os.path.join(input_dir, f"*{fixed_name}*.tl"))[0]

    with open(source_path, "rbU") as f:
        num_lines = sum(1 for _ in f)

    assert num_lines > 2 * sample_size, "Not enough training lines provided"

    valid_idx = np.random.choice(num_lines, sample_size, replace=False)

    source_file = open(source_path, "r")
    target_file = open(target_path, "r")
    src_train_file = open(os.path.join(input_dir, f"train.sl"), "w+")
    tgt_train_file = open(os.path.join(input_dir, f"train.tl"), "w+")
    src_valid_file = open(os.path.join(input_dir, f"valid.sl"), "w+")
    tgt_valid_file = open(os.path.join(input_dir, f"valid.tl"), "w+")

    for i in tqdm(range(num_lines)):
        src_line = source_file.readline()
        tgt_line = target_file.readline()
        if i in valid_idx:
            src_valid_file.write(src_line)
            tgt_valid_file.write(tgt_line)
        else:
            src_train_file.write(src_line)
            tgt_train_file.write(tgt_line)

    src_train_file.close()
    tgt_train_file.close()
    src_valid_file.close()
    tgt_valid_file.close()
    source_file.close()
    target_file.close()

    os.remove(source_path)
    os.remove(target_path)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python split_sets.py <input_dir> <sample_size>")
        sys.exit(1)

    main(input_dir=sys.argv[1], sample_size=int(sys.argv[2]))
