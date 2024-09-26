from typing import Optional, Union, List
import shutil
import torch
import os

from fire import Fire
from transformers import AutoTokenizer, AutoModel, AutoConfig
#from lightning.pytorch import Trainer
#from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
#from lightning.pytorch.tuner import Tuner
import pytorch_lightning as pl
from hf_hub_lightning import HuggingFaceHubCallback

from dataset.modules import ParallelDataModule, LightningTransformer

def train_model(model, tokenizer, engine_name, model_name, batch_size=16, max_length=512, fast_dev_run=False, patience=5, lr=0.0001):
    dm = ParallelDataModule(
		"/opt/ens-dist/runtime/default/tmp/training/datagen/",
		tokenizer,
		max_length=max_length,
		batch_size=batch_size,
		src_pad_idx=tokenizer.pad_token_id,
		tgt_pad_idx=tokenizer.pad_token_id,
		fast_dev_run=fast_dev_run,
	)
    dm.prepare_data()
    dm.setup("fit")
    model = LightningTransformer(model=model, lr=lr, dm=dm)

    early_stopping_callback = pl.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=patience)
    hf_callback = HuggingFaceHubCallback(f"M-Ramo-Translated/{model_name}-{os.getenv('DECODER_DIR')}:lightning")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
		monitor="val_loss",
		mode="min",
		save_top_k=5,
		dirpath=f"checkpoints/{engine_name}/{model_name}/default",
		filename="{global_step}-{val_loss:.2f}"
	)

    trainer = pl.Trainer(
		accelerator="gpu",
		max_epochs=100,
		callbacks=[checkpoint_callback, early_stopping_callback],
		fast_dev_run=fast_dev_run,
		precision="16-mixed",
		log_every_n_steps=1 if fast_dev_run else 50,
		accumulate_grad_batches=4,
	)

    tuner = pl.tuner.Tuner(trainer)
    tuner.lr_find(model)

    trainer.fit(model)
    shutil.copytree("checkpoints", "/result/", dirs_exist_ok=True)
    trainer.push_to_hub(f"M-Ramo-Translated/{model_name}-{os.getenv('DECODER_DIR')}:lightning", use_auth_token=os.getenv("HUGGINGFACE_TOKEN"))
