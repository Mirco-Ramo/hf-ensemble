#!/usr/bin/env bash
set -e
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# add here additional steps to perform
if [[ ! $EXPERIMENT_ID -eq 0 ]]; then
  python3 $SCRIPT_DIR/models/train$EXPERIMENT_ID.py default$EXPERIMENT_ID $TRAINER_STRATEGY
fi

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
if [[ $EXPERIMENT_ID -eq 0 ]]; then
  python3 $SCRIPT_DIR/run_inference.py
fi
# add here additional steps to perform