#!/bin/bash

NAME=$1

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SRC_PATH="${SCRIPT_DIR}/../../.."
MODEL_PATH="${SRC_PATH}/saved_models/${NAME}_model"

python3 timing_curve.py \
    --id "${NAME}_experiment" \
    --save_path "/home/olimoyo/generative-graphik/experiments/experiment_timing/results/" \