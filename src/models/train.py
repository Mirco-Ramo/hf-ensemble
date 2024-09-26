import os
import evaluate
import numpy as np
import wandb
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import EarlyStoppingCallback
from datasets import DatasetDict

from dataset.dataset import ParallelFilesDataset


def print_model_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)

def train_model(model, tokenizer, engine_name, model_name):

    raw_train = ParallelFilesDataset(f"/opt/ens-dist/runtime/default/tmp/training/datagen/train.sl", f"/opt/ens-dist/runtime/default/tmp/training/datagen/train.tl", None).to_hf_dataset()
    raw_valid = ParallelFilesDataset(f"/opt/ens-dist/runtime/default/tmp/training/datagen/valid.sl", f"/opt/ens-dist/runtime/default/tmp/training/datagen/valid.tl", None).to_hf_dataset()
    
    raw_dataset = DatasetDict({"train": raw_train, "validation": raw_valid})

    metric = evaluate.load("sacrebleu")

    def preprocess_function(examples):
        src = [ex["sl"] for ex in examples["translation"]]
        tgt = [ex["tl"] for ex in examples["translation"]]
        model_inputs = tokenizer(src, max_length=512, truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(tgt, max_length=512, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # In case the model returns more than the prediction logits
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Replace -100s in the labels as we can't decode them
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        return {"bleu": result["score"]}

    dataset = raw_dataset.map(preprocess_function, 
                                batched=True, 
                                batch_size=512, 
                                remove_columns=raw_dataset["train"].column_names)

    
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)



    os.environ["WANDB_PROJECT"]=f"{model_name}-{os.getenv('DECODER_DIR')}"

    # save your trained model checkpoint to wandb
    os.environ["WANDB_LOG_MODEL"]="true"

    # turn off watch to log faster
    os.environ["WANDB_WATCH"]="false"

    os.makedirs(f"/opt/ens-dist/engines/{engine_name}/models/decoder/{os.getenv('DECODER_DIR')}", exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        f"/opt/ens-dist/engines/{engine_name}/models/decoder/{os.getenv('DECODER_DIR')}",
        overwrite_output_dir=True,
        #"m2m100_418M-ko",
        seed=314,
        optim="adamw_hf",
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.02,
        save_total_limit=3,
        auto_find_batch_size=True,
        predict_with_generate=True,
        fp16=True,
        do_train=True,
        do_eval=True,
        eval_accumulation_steps=3,
        report_to="wandb",
        metric_for_best_model="bleu",
        load_best_model_at_end = True,
        num_train_epochs=100,
    )

    callback = EarlyStoppingCallback(early_stopping_patience=5)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[callback],
        compute_metrics=compute_metrics,

    )

    trainer.train()
    wandb.finish()
    trainer.push_to_hub(f"M-Ramo-Translated/{model_name}-{os.getenv('DECODER_DIR')}", use_auth_token=os.getenv("HUGGINGFACE_TOKEN"))
