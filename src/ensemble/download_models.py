from transformers import AutoTokenizer, AutoModelForCausalLM, MBartConfig, MBartForConditionalGeneration, M2M100Config, AutoModelForSeq2SeqLM, MarianConfig, MarianMTModel, NllbMoeConfig, NllbMoeForConditionalGeneration
import torch.nn as nn
import os
from tokenizer.train_tokenizer import get_tokenizer_name
from dataset.modules import LightningTransformer
import torch

MODEL_NAMES= ["MBART","M2M100","MarianMTM","NllbMoe"]
MODELS_PATH = "/opt/ens-dist/models/checkpoints"


def get_hf_model(i, tokenizer):
    if i == 1: 
        config = MBartConfig(vocab_size=tokenizer.vocab_size, 
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
                         attention_dropout=0.1
                        )
    
        return MBartForConditionalGeneration(config)

    elif i == 2:
        config = M2M100Config(vocab_size=tokenizer.vocab_size, 
                        decoder_start_token_id=tokenizer.bos_token_id, 
                         encoder_start_token_id=tokenizer.bos_token_id, 
                         eos_token_id=tokenizer.eos_token_id, 
                         pad_token_id=tokenizer.pad_token_id, 
                         decoder_end_token_id=tokenizer.eos_token_id, 
                         encoder_end_token_id=tokenizer.eos_token_id,
                        )
        return AutoModelForSeq2SeqLM.from_pretrained("facebook/m2m100_418M", use_auth_token=os.getenv("HUGGINGFACE_TOKEN"),ignore_mismatched_sizes=True, config=config)
    
    elif i == 3:
        config = MarianConfig(vocab_size=tokenizer.vocab_size, 
                         decoder_start_token_id=tokenizer.bos_token_id, 
                         encoder_start_token_id=tokenizer.bos_token_id, 
                         eos_token_id=tokenizer.eos_token_id, 
                         pad_token_id=tokenizer.pad_token_id, 
                         decoder_end_token_id=tokenizer.eos_token_id, 
                         encoder_end_token_id=tokenizer.eos_token_id,
                         encoder_layers=8,
                         decoder_layers=8,
                         d_model=1024,
                         encoder_ffn_dim=4096,
                         decoder_ffn_dim=4096
                        )

        return MarianMTModel(config).from_pretrained("Helsinki-NLP/opus-mt-en-it", ignore_mismatched_sizes=True, use_auth_token=os.getenv("HUGGINGFACE_TOKEN"))
    elif i == 4:
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

    
        return NllbMoeForConditionalGeneration.from_pretrained("facebook/nllb-moe-54b", ignore_mismatched_sizes=True, use_auth_token=os.getenv("HUGGINGFACE_TOKEN"), config=config)


def download_models():
    """
    Download the models for the ensemble.
    """

    models = []
    all_dirs = str(os.getenv("DATACLEAN_DIRS")).split(";")
    src = all_dirs[0].split("__")[0]
    tgts = [d.split("__")[1] for d in all_dirs]
    tokenizer = AutoTokenizer.from_pretrained(f"{get_tokenizer_name(src, tgts)}", use_auth_token=os.getenv("HUGGINGFACE_TOKEN"))

    for i in range(len(MODEL_NAMES)):
        try:
            model_path = os.path.join(MODELS_PATH, MODEL_NAMES[i])
            hf_model = get_hf_model(i, tokenizer=tokenizer)
            model = LightningTransformer.load_from_checkpoint(os.path.join(model_path, os.listdir(model_path)[0]))
            models.append(model)
        except:
            print(f"Could not download model {MODEL_NAMES[i]}-{os.getenv('DECODER_DIR')}, model not found")
    
    return tokenizer, models
    
