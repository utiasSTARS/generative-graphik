#!/bin/bash

NAME=$1

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SRC_PATH="${SCRIPT_DIR}/../.."
MODEL_PATH="${SRC_PATH}/saved_models/${NAME}_model"

# copy the model and training code if new
if [ -d "${MODEL_PATH}" ]
then
    echo "Directory already exists, using existing model."
else
    echo "Model not found!"
fi

python experiment_accuracy.py \
    --id "${NAME}_experiment" \
    --robots panda \
    --n_evals 500 \
    --model_path "${MODEL_PATH}/" \
    --device cuda:1 \
    --num_samples 32