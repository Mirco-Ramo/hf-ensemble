import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import torch
import math

def ensemble_generate(models, tokenizer, src_text):

    tgt_text=""
    log_probs = []
    models_size = len(models)
    models = nn.ModuleList(models)
    preds = []

    while tokenizer.eos_token not in tgt_text:
        for model in models:
            model_inputs = tokenizer(src_text, text_target=tgt_text, return_tensors="pt")

            probs = model(**model_inputs).logits[:, :, :10]
            probs = F.log_softmax(probs, dim=-1)
            probs = probs[:, -1, :]

            log_probs.append(probs)

        avg_probs = torch.logsumexp(torch.stack(log_probs, dim=0), dim=0) - math.log(models_size)
        last_pred = np.argmax(avg_probs.detach().numpy())
        preds.append(last_pred)
        tgt_text = tokenizer.batch_decode(preds, skip_special_tokens=True)[0]

    return tgt_text