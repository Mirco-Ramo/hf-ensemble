#!/usr/bin/env bash

: '
Entry point
Usage:
  SPECIFIC_VARS=<value> run.sh | tee run.log
  Requires envs/.env and envs/.lang-info to be set
'

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

source $SCRIPT_DIR/env/.env
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
lang_info_file=$SCRIPT_DIR/../env/.lang-info
if [[ ! -z $lang_info_file ]]; then
  source $SCRIPT_DIR/../env/.lang-info
fi
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source $SCRIPT_DIR/src/setup.sh
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source $SCRIPT_DIR/src/run_experiment.sh
