import importlib.util
import json
import os
import time
from collections import OrderedDict
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader

from generative_graphik.args.parser import parse_training_args
from generative_graphik.utils.torch_utils import set_seed_torch
from generative_graphik.utils.dataset_generation import CachedDataset

def _init_fn(worker_id):
    np.random.seed(int(args.random_seed))

def load_datasets(path: str, device, val_pcnt=0):
    with open(path, "rb") as f:
        try:
            # data = pickle.load(f)
            data = torch.load(f)
            data.data = data.data.to(device)
            val_size = int((val_pcnt/100)*len(data))
            train_size = len(data) - val_size
            val_dataset, train_dataset = torch.utils.data.random_split(data, [val_size, train_size])
        except (OSError, IOError) as e:
            val_dataset = None
            train_dataset = None
    return train_dataset, val_dataset

def opt_epoch(paths, model, epoch, device, opt=None, total_batches=100):
    """Single training epoch."""
    if opt:
        model.train()
    else:
        model.eval()

    # Keep track of losses
    running_stats = {}
    total_norm = 0

    num_batches = 0
    with tqdm(total=total_batches) as pbar:
        for path in paths:
            
            # Load training dataset from training path
            all_data, _ = load_datasets(
                path,
                device,
                val_pcnt=0
            )

            loader = DataLoader(
                all_data,
                batch_size=args.n_batch,
                num_workers=args.n_worker,
                shuffle=True,
                worker_init_fn=_init_fn
            )
            for idx, data in enumerate(loader):

                # Pick 1 random config from the samples as the goal and the rest as the random configs
                with torch.no_grad():
                    data_ = model.preprocess(data)

                # Forward call
                res = model.forward(data_)

                # Get loss and stats
                loss, stats = model.loss(res, epoch)

                if opt:
                    opt.zero_grad()
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
                    opt.step()

                num_batches += 1
                pbar.update(1)
                # add stats to running stats
                for key, val in stats.items():
                    if key not in running_stats:
                        running_stats[key] = [val]
                    else:
                        running_stats[key] += [val]
    summary_stats = {f"avg_{k}": sum(v) / len(v) for k, v in running_stats.items()}

    return summary_stats


def train(args):
    # Make directory to save hyperparameters, models, etc
    if not args.debug:
        # Directories based on if using SLURM
        slurm_id = os.environ.get("SLURM_JOB_ID")
        save_dir = os.path.join(args.storage_base_path, args.id)
        if slurm_id is not None:
            user = os.environ.get("USER")
            checkpoint_dir = f"/checkpoint/{user}/{slurm_id}"
        else:
            checkpoint_dir = os.path.join(save_dir, "checkpoints/")

        if not os.path.exists(save_dir + "/hyperparameters.txt"): # if model not yet generated
            print("Saving hyperparameters ...")
            print(args)
            os.makedirs(save_dir, exist_ok=True)
            os.makedirs(checkpoint_dir, exist_ok=True)
            args.__dict__ = OrderedDict(sorted(args.__dict__.items(), key=lambda t: t[0]))
            with open(save_dir + "/hyperparameters.txt", "w") as f:
                json.dump(args.__dict__, f, indent=2)
        else: # if model exists load previous args
            with open(save_dir + "/hyperparameters.txt", "r") as f:
                print("Loading parameters from hyperparameters.txt")
                new_amount_epochs = args.n_epoch
                args.__dict__.update(json.load(f))
                args.n_epoch = new_amount_epochs

        writer = SummaryWriter(log_dir=save_dir)
        tb_data = []

    # Fix random seed
    torch.backends.cudnn.deterministic = args.cudnn_deterministic
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    set_seed_torch(args.random_seed)

    device = torch.device(args.device)

    # Dynamically load the networks module specific to the model
    if args.module_path == "none":
        spec = importlib.util.spec_from_file_location("networks", save_dir + "/model.py")
    else:
        spec = importlib.util.spec_from_file_location("networks", args.module_path)

    network = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(network)

    # Model
    model = network.Model(args).to(device)

    # Optimizer
    params = list(model.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, args.n_scheduler_epoch, gamma=0.5)

    # XXX: If a checkpoint exists, assume preempted and resume training
    initial_epoch = 0
    if not args.debug:
        if os.path.exists(os.path.join(checkpoint_dir, "checkpoint.pth")):
            checkpoint = torch.load(os.path.join(checkpoint_dir, "checkpoint.pth"))
            model.load_state_dict(checkpoint["net"])
            sched.load_state_dict(checkpoint["sched"])
            opt.load_state_dict(checkpoint["opt"])
            initial_epoch = checkpoint["epoch"]
            print(f"Resuming training from checkpoint at epoch {initial_epoch}")

    root = args.training_data_path
    root_val = args.validation_data_path
    paths = [root + "/" + f for f in os.listdir(root) if os.path.isfile(os.path.join(root, f))]
    paths_val = [root_val + "/" + f for f in os.listdir(root_val) if os.path.isfile(os.path.join(root_val, f))]

    # Load one file from the training set to establish total size
    loader = DataLoader(load_datasets(paths[0],device)[0],batch_size=args.n_batch)
    total_batches = len(loader)*len(paths)
    val_loader = DataLoader(load_datasets(paths_val[0],device)[0],batch_size=args.n_batch)
    val_batches = len(val_loader)*len(paths_val)
    del loader, val_loader

    # Training loop
    try:
        for epoch in range(initial_epoch + 1, args.n_epoch + 1):
            tic = time.time()

            # Train for one epoch
            summary_train = opt_epoch(
                paths,
                model=model,
                opt=opt,
                epoch=epoch,
                device=device,
                total_batches=total_batches
            )

            if args.use_validation:
                with torch.no_grad():
                    summary_val = opt_epoch(
                        paths_val,
                        model=model,
                        epoch=epoch,
                        device=device,
                        total_batches=val_batches
                    )
            if sched:
                sched.step()

            epoch_time = time.time() - tic

            print(f"Epoch {epoch}/{args.n_epoch}, Time per epoch: {epoch_time}")
            train_str = "[Train]"
            for key, val in summary_train.items():
                train_str += " " + key + f": {val},"
            print(train_str)

            if args.use_validation:
                val_str = "[Val]"
                for key, val in summary_val.items():
                    val_str += " " + key + f": {val},"

                print(val_str)
            print("----------------------------------------")

            # Store tensorboard data
            if not args.debug:
                for k, v in summary_train.items():
                    tb_data.append((f"train/{k}", v, epoch))

                if args.use_validation:
                    for k, v in summary_val.items():
                        tb_data.append((f"val/{k}", v, epoch))

                if epoch % args.n_checkpoint_epoch == 0:
                    # Write tensorboard data
                    for data in tb_data:
                        writer.add_scalar(data[0], data[1], data[2])
                    tb_data = []

                    # Save model at intermittent checkpoints
                    torch.save(
                        {
                            "net": model.state_dict(),
                            "opt": opt.state_dict(),
                            "sched": sched.state_dict(),
                            "epoch": epoch,
                        },
                        os.path.join(checkpoint_dir, "checkpoint.pth"),
                    )
    finally:
        # Save models
        if not args.debug:
            # Save models
            torch.save(model.state_dict(), save_dir + f"/net.pth")
            writer.close()

if __name__ == "__main__":
    args = parse_training_args()
    train(args)
