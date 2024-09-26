import torch
import datasets


class ParallelFilesDataset(torch.utils.data.Dataset):
    def __init__(self, path_src, path_tgt, tokenizer):
        self.path_src = path_src
        self.path_tgt = path_tgt
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        with open(self.path_src, "r") as f_src, open(self.path_tgt, "r") as f_tgt:
            return f_src.readlines()[index], f_tgt.readlines()[index]

    def __len__(self):
        with open(self.path_src, "r") as f:
            return len(f.readlines())
        
    def to_hf_dataset(self):
        with open(self.path_src, "r") as f_src, open(self.path_tgt, "r") as f_tgt:
            return datasets.Dataset.from_dict({"translation": [{"sl": sl, "tl": tl} for sl, tl in zip(f_src.readlines(), f_tgt.readlines())]})

    

class FileDataset(torch.utils.data.Dataset):
    def __init__(self, path, tokenizer):
        self.path = path
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        with open(self.path_src, "r") as f:
            return self.tokenizer(f.readlines()[index])

    def __len__(self):
        with open(self.path, "r") as f:
            return len(f.readlines())
