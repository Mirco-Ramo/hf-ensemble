import sys
from glob import glob
import random as rnd
import os
import re
import shutil
from tqdm import tqdm

rnd.seed(42)


def main(input_dir: str, output_dir: str, sample_size: int = 10_000) -> None:
    input_paths = glob(os.path.join(input_dir, "*Matecat*"))
    lang_ids = list(set(input_path.rsplit(".", 1)[1] for input_path in input_paths))

    l_paths = glob(os.path.join(input_dir, f"*Matecat*.{lang_ids[0]}"))
    l_sample = rnd.sample(l_paths, sample_size)
    r_sample = [f"{p.rsplit(lang_ids[0], 1)[0]}{lang_ids[1]}" for p in l_sample]
    assert len(l_sample) == len(r_sample) == sample_size, "Sample size mismatch"

    os.makedirs(output_dir, exist_ok=True)
    print(f"Copying {sample_size} files to {output_dir}")

    for l_path, r_path in tqdm(zip(l_sample, r_sample), total=len(l_sample), desc="Copying files"):
        sub_dir = os.path.splitext(os.path.join(output_dir, os.path.basename(l_path)))[0]
        shutil.copyfile(l_path, sub_dir + f"_gold.{lang_ids[0]}")
        shutil.copyfile(r_path, sub_dir + f"_gold.{lang_ids[1]}")


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python split_sets.py <input_dir> <output_dir> <sample_size>")
        sys.exit(1)

    main(input_dir=sys.argv[1], output_dir=sys.argv[2], sample_size=int(sys.argv[3]))