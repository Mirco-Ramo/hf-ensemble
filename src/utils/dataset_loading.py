import numpy as np
import os

def get_training_corpus_length(input, split=None):
        if os.path.isdir(input):
            if split:
                files = [os.path.join(input, f) for f in os.listdir(input) if f.startswith(split)]
            else:
                files = [os.path.join(input, f) for f in os.listdir(input)]
        else:
            files = [input]
        tot_size=0
        for input_file in files:
            with open(input_file, "r") as f:
                tot_size += sum(1 for _ in f)
        return tot_size


def get_training_corpus(input, split=None):
    if os.path.isdir(input):
        if split:
            files = [os.path.join(input, f) for f in os.listdir(input) if f.startswith(split)]
        else:
            files = [os.path.join(input, f) for f in os.listdir(input)]
    else:
        files = [input]
    
    tot_size = get_training_corpus_length(input, split)

    step_size=10000
    for input_file in files:
        with open(input_file, "r") as f:
            for start_idx in range(0, tot_size, step_size):
                lines = []
                for i in range(start_idx, start_idx + step_size):
                    lines.append(f.read())
                yield lines


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels




