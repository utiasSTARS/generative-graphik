#!/bin/bash

NAME=$1

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SRC_PATH="${SCRIPT_DIR}/../.."
MODEL_PATH="${SRC_PATH}/saved_models/paper_models/${NAME}_model"

export PYTORCH_ENABLE_MPS_FALLBACK=1
# copy the model and training code if new
if [ -d "${MODEL_PATH}" ]
then
    echo "Directory already exists, using existing model."
else
    echo "Model not found!"
fi

python experiment_2.py \
    --id "${NAME}_experiment" \
    --robots panda kuka lwa4d lwa4p ur10 \
    --n_evals 1000 \
    --model_path "${MODEL_PATH}/" \
    --device cpu \
    --num_samples 32
