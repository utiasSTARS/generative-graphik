#!/bin/bash

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SRC_PATH="${SCRIPT_DIR}/.."

NAME=$1
DATASET_NAME=$2
VALIDATION_DATASET_NAME=${3:-"${DATASET_NAME}_validation"}
TRAIN=${4:-true}

MODEL_PATH="${SRC_PATH}/saved_models/${NAME}_model"

# export PYTORCH_ENABLE_MPS_FALLBACK=1

# copy the model and training code if new
if [ -d "${MODEL_PATH}" ]
then
    echo "Directory already exists, using existing model."
else
    echo "Creating new model directory."

    mkdir ${MODEL_PATH}
    cp ${SRC_PATH}/generative_graphik/model.py ${MODEL_PATH}/model.py
fi

if [ ! -d "${SRC_PATH}/datasets/${DATASET_NAME}" ]
then
    echo "Dataset ${DATASET_NAME} not found, creating it."
    python -u ${SRC_PATH}/generative_graphik/utils/dataset_generation.py \
        --id "${DATASET_NAME}" \
        --robots ur10 \
        --num_examples 100000 \
        --max_examples_per_file 100000 \
        --goal_type pose \
        --randomize False
else
    echo "Dataset ${DATASET_NAME} found!"
fi

if [ "${TRAIN}" = true ]
then
python -u ${SRC_PATH}/generative_graphik/train.py \
    --id "${NAME}_model" \
    --norm_layer LayerNorm \
    --debug False \
    --device cuda \
    --n_worker 0 \
    --n_beta_scaling_epoch 1 \
    --lr 3e-4 \
    --n_batch 128 \
    --num_graph_mlp_layers 2 \
    --num_egnn_mlp_layers 2 \
    --graph_mlp_hidden_size 128 \
    --mlp_hidden_size 128 \
    --dim_latent_node_out 16 \
    --dim_latent 128 \
    --gnn_type "egnn" \
    --num_gnn_layers 5 \
    --num_node_features_out 3 \
    --num_coordinates_in 3 \
    --num_features_in 3 \
    --num_edge_features_in 1 \
    --num_prior_mixture_components 16 \
    --num_likelihood_mixture_components 1\
    --num_anchor_nodes 4 \
    --train_prior True \
    --n_epoch 1 \
    --n_scheduler_epoch 60\
    --dim_goal 6 \
    --storage_base_path "${SRC_PATH}/saved_models" \
    --training_data_path "${SRC_PATH}/datasets/${DATASET_NAME}" \
    --validation_data_path "${SRC_PATH}/datasets/${VALIDATION_DATASET_NAME}" \
    --module_path "${SRC_PATH}/generative_graphik/model.py" \
    --use_validation True \
    --n_checkpoint_epoch 16 \
    --non_linearity silu \
    --rec_gain 10
fi