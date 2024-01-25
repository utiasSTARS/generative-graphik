#!/bin/bash

NAME=$1

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SRC_PATH="${SCRIPT_DIR}/../.."
MODEL_PATH="${SRC_PATH}/saved_models/paper_models/${NAME}_model"

python table.py \
    --id "${NAME}_experiment" \
    --save_latex True \
    --latex_path "/home/filipmrc/Documents/Latex/2022-limoyo-maric-generative-corl/" \

python dist_images.py \
    --id "${NAME}_experiment" \
    --robots ur10 \
    --n_evals 10 \
    --model_path "${MODEL_PATH}/" \
    --device cpu \
    --randomize False \
    --num_samples 32
