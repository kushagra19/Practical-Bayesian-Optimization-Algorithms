import torch
import argparse

from experiments import Experiment

parser = argparse.ArgumentParser(description='Script for BO experiments')

# Experiment Tracking
parser.add_argument('-e','--experiment_name', type=str, required=True,
    help='''Name of experiment corresponding to which experiment directory
        will be created''')
parser.add_argument('-ed', '--exp_dir', type=str, default='./',
    help='Directory where the experiment directory will be created')

# Objective Function
parser.add_argument('-obj_fn', '--objective_function', type=str, required=True,
    choices=['le_branke', 'branin','holder_table', 'levy', 'rosenbrock', 'ackley', 'hartmann', 'powell','gw','Cosine8','Drop_Wave' \
    ,'StyblinskiTang','DixonPrice','SixHumpCamel','ThreeHumpCamel','Sphere','Bukin','Griewank','Michalewicz','Sum_exp','cart_pole','illustrationNd','lunar_lander'\
    'rotation_transformer'],
    help='Objective function to maximize')
parser.add_argument('-m', '--max/min', type=str, default='max',
    choices=['max', 'min'],
    help='whether to maximize or minimize the objective function')
parser.add_argument('-d', '--dims', type=int, required=True,
    help='Dimension of the input to objective function')

# BO Algorithm
parser.add_argument('-zacq', '--zobo_acq_func', type=str, default='EI',
    help='''Acquisition function for ZOBO. Also required if using point
        suggested by function GP''')
parser.add_argument('-o', '--order', type=int, default=0, choices=[0, 1],
    help='If equal to 0 then ZOBO. If equal to 1 then FOBO')
parser.add_argument('-q', '--query_point_selection', type=str,
    default='topk_convex',
    choices=['convex', 'maximum', 'best', 'topk', 'topk_convex'],
    help='Method to be used for query point selection for FOBO')
parser.add_argument('-gacq', '--grad_acq_func', type=str,
    default='SumGradientAcquisitionFunction',
    choices=['PrabuAcquistionFunction', 'SumGradientAcquisitionFunction'],
    help='Type of grad acquisition function to use')

# BO Hyperparams
parser.add_argument('-b', '--budget', type=int, default=100,
    help='Number of function evaluations allowed in one run')
parser.add_argument('-r', '--runs', type=int, default=20,
    help='Number of independent runs over which the results will be averaged')
parser.add_argument('-nm', '--noise_mean', type=float, default=0.0,
    help='Mean of the normal noise to be added to function evaluations')
parser.add_argument('-nv', '--noise_variance', type=float, default=0.01,
    help='Variance of the normal noise to be added to function evaluations')
parser.add_argument('-nr', '--num_restarts', type=int, default=10,
    help='Number of restarts for acquisition function optimisation')
parser.add_argument('-rs', '--raw_samples', type=int, default=32,
    help='Number of raw samples for acquisition function optimisation')
parser.add_argument('-s', '--seed', type=int, default=42, help='Random seed')
parser.add_argument('-i', '--init_examples', type=int, default=5,
    help='Number of initial data samples to be used')
args = parser.parse_args()

config = vars(args)
print(config)
dtype = torch.double

experiment = Experiment(is_major_change=False, config=config, dtype=dtype)
experiment.multi()
