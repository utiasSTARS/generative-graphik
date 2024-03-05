#!/bin/bash
python experiment_infeasible_poses.py \
    --robots kuka panda ur10 lwa4d lwa4p \
    --model_paths /home/olimoyo/generative-graphik/saved_models/paper_models/kuka_512k_model /home/olimoyo/generative-graphik/saved_models/paper_models/panda_512k_model /home/olimoyo/generative-graphik/saved_models/paper_models/ur10_512k_model /home/olimoyo/generative-graphik/saved_models/paper_models/lwa4d_512k_model /home/olimoyo/generative-graphik/saved_models/paper_models/lwa4p_512k_model \
    --infeasible_pose_paths /home/olimoyo/generative-graphik/datasets/infeasible_poses/infeasible_poses_kuka.pkl /home/olimoyo/generative-graphik/datasets/infeasible_poses/infeasible_poses_panda.pkl /home/olimoyo/generative-graphik/datasets/infeasible_poses/infeasible_poses_ur10.pkl /home/olimoyo/generative-graphik/datasets/infeasible_poses/infeasible_poses_lwa4d.pkl /home/olimoyo/generative-graphik/datasets/infeasible_poses/infeasible_poses_lwa4p.pkl\
    --dataset_paths /media/stonehenge/users/oliver-limoyo/2.56m-kuka /media/stonehenge/users/oliver-limoyo/2.56m-panda /media/stonehenge/users/oliver-limoyo/2.56m-ur10 /media/stonehenge/users/oliver-limoyo/2.56m-lwa4d /media/stonehenge/users/oliver-limoyo/2.56m-lwa4p \
    --device cuda:1 \
