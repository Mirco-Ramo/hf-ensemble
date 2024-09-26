from dataset.dataset import FileDataset
from ensemble.download_models import download_models
from ensemble.ensemble_generate import ensemble_generate
import yaml
import os
import shutil

def main():
   tokenizer, models = download_models() 

   with open("/opt/ens-dist/configs/datasets.yaml", "r") as f:
      datasets = yaml.load(f, Loader=yaml.FullLoader)

   direction = os.getenv("TRANSLATION_DIR")
   dir_datasets = datasets[direction]
   for dataset in dir_datasets:
      src_dataset = dataset["src"]
      try:
         ref_dataset = dataset["ref"]
      except KeyError:
         ref_dataset = None
      shutil.copy(src_dataset, f"runtime/default/translation/{direction}/{dataset}/source.txt")
      shutil.copy(ref_dataset, f"runtime/default/translation/{direction}/{dataset}/reference.txt") if ref_dataset else None
   
      with open(src_dataset, "r") as f_src, open(f"runtime/default/translation/{direction}/{dataset}/hypothesis.txt") as f_hyp:
         for line in f_src.readlines():
            f_hyp.write(ensemble_generate(models, tokenizer, line))




if __name__ == "main":
    main()