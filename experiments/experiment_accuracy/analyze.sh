#!/bin/bash

NAME=$1

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SRC_PATH="${SCRIPT_DIR}/../../.."
MODEL_PATH="${SRC_PATH}/saved_models/${NAME}_model"

python table.py \
    --id "${NAME}_experiment" \
    --save_latex True \
    --latex_path "/home/olimoyo/generative-graphik/experiments/experiment_accuracy/results/"
