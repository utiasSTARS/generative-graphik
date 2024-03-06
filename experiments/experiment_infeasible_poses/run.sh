#!/bin/bash
python experiment_infeasible_poses.py \
    --robots kuka panda ur10 lwa4d lwa4p \
    --model_paths /home/olimoyo/generative-graphik/saved_models/512k-kuka_model  \
    --infeasible_pose_paths /home/olimoyo/generative-graphik/datasets/infeasible_poses/infeasible_poses_kuka.pkl \
    --device cuda:0 \
