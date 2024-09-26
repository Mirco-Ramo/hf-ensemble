"""
Train a MBart for translation

Usage: python train1.py <engine_name> 

Example: python train1.py "ensemble"
"""

from transformers import NllbMoeForConditionalGeneration, AutoTokenizer, NllbMoeConfig

import sys
import os

from tokenizer.train_tokenizer import get_tokenizer_name
from models.train import print_model_params


def main(engine_name, trainer="hf"):

    if trainer == "hf":
        from models.train import train_model
    else:
        from models.lightning_train import train_model

    all_dirs = os.getenv("DATACLEAN_DIRS").split(";")
    src = all_dirs[0].split("__")[0]
    tgts = [d.split("__")[1] for d in all_dirs]

    tokenizer = AutoTokenizer.from_pretrained(get_tokenizer_name(src, tgts), use_auth_token=os.getenv("HUGGINGFACE_TOKEN"))
    tokenizer.src_lang = os.getenv("TRANSLATION_DIR").split("__")[0]
    tokenizer.tgt_lang = os.getenv("TRANSLATION_DIR").split("__")[1]

    config = NllbMoeConfig(vocab_size=tokenizer.vocab_size, 
                         decoder_start_token_id=tokenizer.bos_token_id, 
                         encoder_start_token_id=tokenizer.bos_token_id, 
                         eos_token_id=tokenizer.eos_token_id, 
                         pad_token_id=tokenizer.pad_token_id, 
                         decoder_end_token_id=tokenizer.eos_token_id, 
                         encoder_end_token_id=tokenizer.eos_token_id,
                         max_position_embeddings=512,
                         encoder_layers=6,
                         encoder_ffn_dim=4096,
                         encoder_attention_heads=16,
                         decoder_layers=6,
                         decoder_ffn_dim=4096,
                         decoder_attention_heads=16,
                         d_model=1024,
                         dropout=0.2,
                         attention_dropout=0.1,
                         num_experts=16,
                         expert_capacity=64
                        )

    
    model = NllbMoeForConditionalGeneration.from_pretrained("facebook/nllb-moe-54b", ignore_mismatched_sizes=True, use_auth_token=os.getenv("HUGGINGFACE_TOKEN"), config=config)
    print_model_params(model)
    train_model(model, tokenizer, engine_name, "NllbMoe")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "hf")