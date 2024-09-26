import os
import sys

from utils.dataset_loading import get_training_corpus, get_training_corpus_length

from transformers import AutoTokenizer
from tokenizers.processors import TemplateProcessing

def get_tokenizer_name(src_lang, target_languages):
    language_code = "_".join(target_languages)
    if len(target_languages)>1:
        return f"M-Ramo-Translated/tokenizer-{src_lang}__{language_code}"
    else:
        return f"M-Ramo-Translated/tokenizer-{src_lang}__{target_languages[0]}"


def train_tokenizer(input_file, vocab_size, src_lang, target_languages):

    language_code = "_".join(target_languages)

    try:
        tokenizer = AutoTokenizer.from_pretrained(f"{get_tokenizer_name(src_lang, target_languages)}", use_auth_token=os.getenv("HUGGINGFACE_TOKEN"))
        if not tokenizer:
            raise Exception("Tokenizer not found")
        print("Tokenizer already trained")
        return tokenizer
    except:
        print("Training tokenizer")

    old_tokenizer = AutoTokenizer.from_pretrained("gpt2")


    training_corpus = get_training_corpus(input_file, "train")

    if len(target_languages) > 1:
        tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, vocab_size=vocab_size, show_progress=True, length=get_training_corpus_length(input_file, "train"), new_special_tokens=[f"<{l}>" for l in language_code]+["<s>", "</s>", "<pad>"], special_tokens_map={"bos_token": "<s>", "eos_token": "</s>", "pad_token": "<pad>"})

    else:
        tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, vocab_size=vocab_size, show_progress=True, length=get_training_corpus_length(input_file, "train"), new_special_tokens=["<s>", "</s>", "<pad>"], special_tokens_map={"bos_token": "<s>", "eos_token": "</s>", "pad_token": "<pad>"})

    return tokenizer



def main(input_file, vocab_size, directions):

    directions = directions.split(";")
    src = directions[0].split("__")[0]  
    tgts = [d.split("__")[1] for d in directions]

    tokenizer = train_tokenizer(input_file, vocab_size, src, tgts)
    tokenizer.bos_token = "<s>"
    tokenizer.eos_token = "</s>"
    tokenizer.pad_token = "<pad>"
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=tokenizer.bos_token + " $A " + tokenizer.eos_token,
        special_tokens=[(tokenizer.eos_token, tokenizer.eos_token_id), (tokenizer.bos_token, tokenizer.bos_token_id)],
    )

    tokenizer.push_to_hub(get_tokenizer_name(src, tgts), use_auth_token=os.environ["HUGGINGFACE_TOKEN"], private=True)

if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3]) 