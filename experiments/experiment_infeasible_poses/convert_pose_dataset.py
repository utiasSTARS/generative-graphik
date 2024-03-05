import argparse
import os 

import numpy as np
import pickle

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader

def load_datasets(path: str, device, val_pcnt=0):
    with open(path, "rb") as f:
        try:
            # data = pickle.load(f)
            data = torch.load(f)
            data._data = data._data.to(device)
            val_size = int((val_pcnt/100)*len(data))
            train_size = len(data) - val_size
            val_dataset, train_dataset = torch.utils.data.random_split(data, [val_size, train_size])
        except (OSError, IOError) as e:
            val_dataset = None
            train_dataset = None
    return train_dataset, val_dataset

class CachedDataset(InMemoryDataset):
    def __init__(self, data, slices):
        super(CachedDataset, self).__init__(None)
        self.data, self.slices = data, slices

def convert_poses(args):
    dataset_path = os.path.join(args.dataset_path, "data_0.p")
    # with open(os.path.join(args.dataset_path, "data_0.p"), 'rb') as f:
    #     reference_dataset = torch.load(f)

    # Load training dataset from training path
    all_data, _ = load_datasets(
        dataset_path,
        "cpu",
        val_pcnt=0
    )

    loader = DataLoader(
        all_data,
        batch_size=256,
        num_workers=16,
        shuffle=False    
    )
    pose_list = []
    for idx, data in enumerate(loader):
        pose_list.append(data.T_ee.cpu().detach().numpy())
    pose_list = np.concatenate(pose_list)

    with open(os.path.join(args.dataset_path, f"np_poses.pkl"), 'wb') as f:
        # Dump the list of NumPy arrays into the file
        pickle.dump(pose_list, f)

def parse_convert_poses_args():
    parser = argparse.ArgumentParser()

    # General settings
    parser.add_argument("--dataset_path", type=str, default="/media/stonehenge/users/oliver-limoyo/2.56m-lwa4p", help="Path to folder with infeasible poses to test with.")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_convert_poses_args()
    infeasible_poses = convert_poses(args)
