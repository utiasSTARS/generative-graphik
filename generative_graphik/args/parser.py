import argparse

from generative_graphik.args.utils import str2bool, str2inttuple, str2tuple, str2floattuple

def parse_training_args():
    parser = argparse.ArgumentParser()

    # Experiment Settings
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for PyTorch')
    parser.add_argument('--cudnn_deterministic', type=str2bool, default=True, help='Use cudnn deterministic')
    parser.add_argument('--cudnn_benchmark', type=str2bool, default=False, help='Use cudnn benchmark')
    parser.add_argument('--id', type=str, default="None", help='Name of folder used to store model')
    parser.add_argument('--random_seed', type=int, default=333, help='Random seed')
    parser.add_argument('--use_validation', type=str2bool, default=True, help='Run validation')
    parser.add_argument('--debug', type=str2bool, default=True, help='Debug and do not save models or log anything')
    parser.add_argument('--storage_base_path', type=str, required=True, help='Base path to store all training data')
    parser.add_argument('--training_data_path', type=str, default="planar_chain_size_20000_dof_[5, 7, 9, 11]_randomize_True_partial_True_approx_edges_False_data.p", help='Path to training data')
    parser.add_argument('--validation_data_path', type=str, default="planar_chain_size_20000_dof_[5, 7, 9, 11]_randomize_True_partial_True_approx_edges_False_data.p", help='Path to training data')
    parser.add_argument('--module_path', type=str, default="none", help='Path to network module.')

    # Training Settings
    parser.add_argument('--n_epoch', type=int, default=4096, help='Number of epochs')
    parser.add_argument('--n_scheduler_epoch', type=int, default=25, help='Number of epochs before fixed scheduler steps.')
    parser.add_argument('--n_checkpoint_epoch', type=int, default=32, help='Number of epochs for checkpointing')
    parser.add_argument('--n_beta_scaling_epoch', type=int, default=1, help='Warm start KL divergence for this amount of epochs.')
    parser.add_argument('--n_joint_scaling_epoch', type=int, default=1, help='Warm start joint loss for this amount of epochs.')
    parser.add_argument('--n_batch', type=int, default=32,  help='Batch size')
    parser.add_argument('--n_worker', type=int, default=16,  help='Amount of workers for dataloading.')
    parser.add_argument('--lr', type=float, default= 3e-4, help='Learning rate')

    # Network parameters
    parser.add_argument('--num_anchor_nodes', type=int, default=3, help='Number of anchor nodes')
    parser.add_argument('--num_node_features_out', type=int, default=3,  help='Size of node features out')
    parser.add_argument('--num_coordinates_in', type=int, default=3,  help='Size of node coordinates in')
    parser.add_argument('--num_features_in', type=int, default=3,  help='Size of node features in')
    parser.add_argument('--num_edge_features_in', type=int, default=1,  help='Size of edge features in')
    parser.add_argument('--gnn_type', type=str, default="egnn", help='GNN type used.')
    parser.add_argument('--num_gnn_layers', type=int, default=3,  help='Number of GNN layers')
    parser.add_argument('--num_graph_mlp_layers', type=int, default=0,  help='Number of layers for the MLPs used in the graph')
    parser.add_argument('--num_egnn_mlp_layers', type=int, default=2,  help='Number of layers for the MLPs used in the EGNN layer itself')
    parser.add_argument('--num_iterations', type=int, default=1,  help='Number of iterations to networks go through')
    parser.add_argument('--dim_latent', type=int, default=8,  help='Size of latent node features in to encoder')
    parser.add_argument('--dim_goal', type=int, default=3,  help='Size of goal representation (SE3-->6, SE2-->3)')
    parser.add_argument('--num_prior_mixture_components', type=int, default=1, help='Number of mixture components for prior network')
    parser.add_argument('--num_likelihood_mixture_components', type=int, default=1, help='Number of mixture components for likelihood network')
    parser.add_argument('--train_prior', type=str2bool, default=True, help='Learn prior parameters conditionned on variables.')
    parser.add_argument('--rec_gain', type=int, default=80,  help='Gain on non-anchor node reconstruction')
    parser.add_argument('--non_linearity', type=str, default="silu", help='Non-linearity used.')
    parser.add_argument('--dim_latent_node_out', type=int, default=3,  help='Size of node feature dim in enc/dec')
    parser.add_argument('--graph_mlp_hidden_size', type=int, default=4,  help='Size of hiddden layers of MLP used in GNN')
    parser.add_argument('--mlp_hidden_size', type=int, default=4,  help='Size of all other MLP hiddden layers')
    parser.add_argument('--norm_layer', choices=['None', 'BatchNorm', 'LayerNorm', 'GroupNorm', 'InstanceNorm', 'GraphNorm'], default='None', help='Layer normalization method.')

    args = parser.parse_args()
    return args

def parse_data_generation_args():
    parser = argparse.ArgumentParser()

    # General settings
    parser.add_argument("--id", type=str, default=None, help="Name of the dataset")
    parser.add_argument('--storage_base_path', type=str, default=None,  help='Base path for storing dataset')

    # Robot settings
    parser.add_argument("--robots", nargs="*", type=str, default=["planar_chain"], help="Type of robot used")
    parser.add_argument("--dofs", nargs="*", type=int, default=[5,7,9,11], help="Numbers of DoF that occur in the dataset")
    parser.add_argument("--randomize", type=str2bool, default=True, help="Randomize kinematic parameters for every instance")
    parser.add_argument("--randomize_percentage", type=float, default=0.2, help="Percentage variation of link lengths.")
    parser.add_argument("--goal_type", type=str, default="pose", help="Randomize kinematic parameters for every instance")
    parser.add_argument("--obstacles", type=str2bool, default=False, help="Use obstacles")
    parser.add_argument("--semantic", type=str2bool, default=False, help="Use semantic tags for nodes.")
    parser.add_argument("--load_from_disk", type=str2bool, default=False, help="Save data as separate files.")

    # Problem settings
    parser.add_argument("--num_examples", type=int, default=10, help="Total number of problems in the dataset")
    parser.add_argument("--validation_percentage", type=int, default=10, help="Percentage of validation data")
    parser.add_argument("--num_samples", type=int, default=100, help="Total number of samples per problem")
    parser.add_argument("--max_examples_per_file", type=int, default=512, help="Max number of problems per file in the dataset")

    args = parser.parse_args()
    return args

def parse_analysis_args():
    parser = argparse.ArgumentParser()

    # General settings
    parser.add_argument("--id", type=str, default="test_experiment", help="Name of the folder with experiment data")
    parser.add_argument('--storage_base_path', type=str, default=None, help='Base path for folder with experiment data')
    parser.add_argument("--model_path", nargs="*", type=str, required=True, help="Path to folder with model data")
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for PyTorch')
    parser.add_argument("--num_samples", nargs="*", type=int, default=1, help="Number of samples to generate")

    # Robot settings
    parser.add_argument("--robots", nargs="*", type=str, default=["planar_chain"], help="Numbers of DoF that are validated")
    parser.add_argument("--dofs", nargs="*", type=int, default=[6,8,10,12], help="Numbers of DoF that are validated")
    parser.add_argument("--n_evals", type=int, default=100, help="Number of evaluations")
    parser.add_argument("--randomize", type=str2bool, default=True, help="Randomize link lengths during test time.")

    args = parser.parse_args()
    return args
