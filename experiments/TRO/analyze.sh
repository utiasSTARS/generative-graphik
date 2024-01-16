#!/bin/bash

NAME=$1

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SRC_PATH="${SCRIPT_DIR}/../.."
MODEL_PATH="${SRC_PATH}/saved_models/paper_models/${NAME}_model"

python table.py \
    --id "${NAME}_experiment" \
    --save_latex True \
    --latex_path "/home/filipmrc/Documents/Latex/2022-limoyo-maric-generative-corl/" \

# python waterfall.py \
#     --id "${NAME}_experiment" \
#     --save_latex False \
#     --latex_path "/home/filipmrc/Documents/Latex/2022-limoyo-maric-generative-corl/"

# python error_bars.py \
#     --id "${NAME}_experiment" \
#     --save_latex False \
#     --latex_path "/home/filipmrc/Documents/Latex/2022-limoyo-maric-generative-corl/"

# python timings.py \
#     --id "${NAME}_experiment" \
#     --robots kuka \
#     --n_evals 100 \
#     --model_path "${MODEL_PATH}/" \
#     --device cuda:0 \
#     --num_samples 10 20 30 40 50 60 70 80 90 100 110 120 130 140

# python dist_images.py \
#     --id "${NAME}_experiment" \
#     --robots panda\
#     --n_evals 10 \
#     --model_path "${MODEL_PATH}/" \
#     --device cpu \
#     --randomize False \
#     --num_samples 32
