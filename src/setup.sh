#!/usr/bin/env bash

set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# define variables to fetch data from S3 buckets
arr_direction=(${TRANSLATION_DIR//__/ })
src_lang=${arr_direction[0]}
tgt_lang=${arr_direction[1]}

# generic engine configurations
ENGINE_NAME="default"

BPE_TOKENS=32000

if [[ -z "$AWS_ACCESS_KEY_ID" || -z "$AWS_SECRET_ACCESS_KEY" ]]; then
    echo "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY should be set." >&2
    exit 1
fi

IFS=, read -ra DATACLEAN_VERSIONS_ARR <<< "$DATACLEAN_VERSIONS"
IFS=, read -ra DATACLEAN_DIRS_ARR <<< "$DATACLEAN_DIRS"

mkdir -p runtime/$ENGINE_NAME/tmp/training/datagen/
# add direction specific datagen
for  (( i=0; i<${#DATACLEAN_DIRS_ARR[*]}; ++i)); do
    aws s3 cp s3://modernmt.prod.cookingbot/data/clean/${DATACLEAN_DIRS_ARR[$i]}/${DATACLEAN_VERSIONS_ARR[$i]}/${DATACLEAN_DIRS_ARR[$i]}.tar.gz runtime/$ENGINE_NAME/tmp/training/data_clean/${DATACLEAN_DIRS_ARR[$i]}.tar.gz
    tar -xf runtime/$ENGINE_NAME/tmp/training/data_clean/${DATACLEAN_DIRS_ARR[$i]}.tar.gz -C runtime/$ENGINE_NAME/tmp/training/data_clean
    find runtime/$ENGINE_NAME/tmp/training/data_clean/${DATACLEAN_DIRS_ARR[$i]} -type f -name "*" -print0 | xargs -0 mv -t runtime/$ENGINE_NAME/tmp/training/data_clean
    rm -rf runtime/$ENGINE_NAME/tmp/training/data_clean/${DATACLEAN_DIRS_ARR[$i]}*
    cur_dir=${DATACLEAN_DIRS_ARR[$i]}
    arr_direction=(${cur_dir//__/ })
    src_lang=${arr_direction[0]}
    tgt_lang=${arr_direction[1]}

    mkdir -p runtime/$ENGINE_NAME/tmp/training/temp
    echo runtime/$ENGINE_NAME/tmp/training/data_clean/*.$src_lang | xargs cat > runtime/$ENGINE_NAME/tmp/training/temp/all_$src_lang.txt
    echo runtime/$ENGINE_NAME/tmp/training/data_clean/*.$tgt_lang | xargs cat > runtime/$ENGINE_NAME/tmp/training/temp/all_$tgt_lang.txt
    find runtime/$ENGINE_NAME/tmp/training/data_clean/ ! -name train* -type f -exec rm -f {} +
    mv runtime/$ENGINE_NAME/tmp/training/temp/all_$src_lang.txt runtime/$ENGINE_NAME/tmp/training/data_clean/dataset$i.$src_lang
    mv runtime/$ENGINE_NAME/tmp/training/temp/all_$tgt_lang.txt runtime/$ENGINE_NAME/tmp/training/data_clean/dataset$i.$tgt_lang
    rm -rf runtime/$ENGINE_NAME/tmp/training/temp
done

#add language identifier for multilingual datagens
if [[ -f runtime/$ENGINE_NAME/tmp/training/data_clean/dataset1.$src_lang ]]; then
    for  (( i=0; i<${#DATACLEAN_DIRS_ARR[*]}; ++i)); do
        cur_dir=${DATACLEAN_DIRS_ARR[$i]}
        arr_direction=(${cur_dir//__/ })
        src_lang=${arr_direction[0]}
        tgt_lang=${arr_direction[1]}
        awk '$0="<$tgt_lang> "$0' runtime/$ENGINE_NAME/tmp/training/data_clean/dataset$i.$src_lang > runtime/$ENGINE_NAME/tmp/training/data_clean/dataset$i.$src_lang 
    done
fi

for  (( i=0; i<${#DATACLEAN_DIRS_ARR[*]}; ++i)); do
    cur_dir=${DATACLEAN_DIRS_ARR[$i]}
    arr_direction=(${cur_dir//__/ })
    src_lang=${arr_direction[0]}
    tgt_lang=${arr_direction[1]}
    cat runtime/$ENGINE_NAME/tmp/training/data_clean/dataset$i.$src_lang >> runtime/$ENGINE_NAME/tmp/training/datagen/dataset.sl
    cat runtime/$ENGINE_NAME/tmp/training/data_clean/dataset$i.$tgt_lang >> runtime/$ENGINE_NAME/tmp/training/datagen/dataset.tl
done


echo "creating train and validation sets"
python3 $SCRIPT_DIR/split_sets.py runtime/$ENGINE_NAME/tmp/training/datagen/  3000

export PYTHONPATH="${PYTHONPATH}:/opt/ens-dist/src"
python3 $SCRIPT_DIR/tokenizer/train_tokenizer.py runtime/$ENGINE_NAME/tmp/training/datagen/  $BPE_TOKENS $DATACLEAN_DIRS_ARR

  


