# generative-graphIK
Code for paper on [Generative Graphical Inverse Kinematics](https://arxiv.org/abs/2209.08812)

## Installation
Install matching versions of [PyTorch](https://pytorch.org/get-started/previous-versions/#v1101:~:text=org/whl/cpu-,v1.10.1,-Conda) and [PyTorch-Geometric](https://pytorch-geometric.readthedocs.io/en/2.0.3/notes/installation.html). We used `torch-1.10.1` and `torch-geometric-2.0.4` but other versions should work as well. 

After installing the above:
```
pip install -e .
```

## Generate a dataset and train
```
./train.sh <yourmodelname> <yourdatasetname>
```

See `./generative-graphik/generative_graphik/args/parser.py` for more details on data generation and model parameters.

## Modifying data generation
To modify the training data, modify lines 29-35 of `train.sh`.

To train on specific robots:
```
    python -u ${SRC_PATH}/generative_graphik/utils/dataset_generation.py \
        --id "${DATASET_NAME}" \
        --robots ur10 kuka panda lwa4d lwa4p \
        --num_examples 512000 \
        --max_examples_per_file 512000 \
        --goal_type pose \
        --randomize False
```

To train on random robots of DOFs 6 and 7:
```
    python -u ${SRC_PATH}/generative_graphik/utils/dataset_generation.py \
        --id "${DATASET_NAME}" \
        --robots revolute_chain \
        --dof 7 6 \
        --num_examples 512000 \
        --max_examples_per_file 512000 \
        --goal_type pose \
        --randomize True \
        --randomize_percentage 0.4
```

## Modifying hyperparameters
To modify the model parameters, modify lines 43-77 of `train.sh`.
