import os
import glob
import numpy as np
import yaml
import torch
import argparse
import matplotlib.pyplot as plt

from experiments import Experiment

def build_obj_func(config : dict):
    experiment = Experiment(is_major_change=False, config=config,
        dtype=torch.double, mode='eval')

    return experiment.obj_fn

def combine_output(config, exp_path, pattern='y_*.pt', out_name='y.pt'):
    # ys_path = glob.glob(os.path.join(exp_path, pattern))

    ys_path = sorted( filter( os.path.isfile,glob.glob(os.path.join(exp_path, pattern)) ) )

    # Xs_path = glob.glob(os.path.join(exp_path, 'X_*.pt'))

    Xs_path = sorted( filter( os.path.isfile,glob.glob(os.path.join(exp_path, 'X_*.pt')) ) )
    base_size = config["budget"] + config["init_examples"]
    y = torch.empty((
        base_size,
        1,
        config["runs"]
    ))

    y_true = torch.empty((
        base_size,
        1,
        config["runs"]
    ))

    X = torch.empty((
            base_size,
            config["dims"],
            config["runs"]
        ))

    for i,y_path in enumerate(ys_path):
        # print(y_path)
        y[:, :, i] = torch.load(y_path).unsqueeze(dim=-1)

    for i, X_path in enumerate(Xs_path):
        # print(X_path)
        X[:, :, i] = torch.load(X_path)

    torch.save(y, os.path.join(exp_path, out_name))
    torch.save(X, os.path.join(exp_path, 'X.pt'))

    return y,X,y_true


def process_output(y,X,y_true,obj_fn, init_examples, mode='max'):
    # diff = torch.abs(y - obj_fn.true_opt_value)
    # print(y)
    if mode=='min':
        cum_res, indices = torch.cummin(y, dim=0)
        # print(cum_res)
        # print(X[indices[:,:,1].squeeze(dim=-1),:,1])
        # print(obj_fn.forward_true(X[indices[:,:,100].squeeze(dim=-1),:,100]))
        # print(indices[:,:,0].squeeze(dim=-1))
    elif mode=='max':
        cum_res, indices = torch.cummax(y, dim=0)
    else:
        raise NotImplementedError

    for i in range(indices.shape[2]):
        # print(i)
        y_true[:,:,i] = obj_fn.forward_true(X[indices[:,:,i].squeeze(dim=-1),:,i]).unsqueeze(dim=-1)
        # print(obj_fn.forward_true(X[indices[:,:,i].squeeze(dim=-1),:,i]).unsqueeze(dim=-1))
        # print(X[indices[:,:,i].squeeze(dim=-1),:,i].shape)
    
    # print(y_true)
    # y_true = cum_res
    regret = torch.log10(torch.abs(y_true-obj_fn.true_opt_value))
    mean_regret = regret.mean(dim=-1).detach()[init_examples:, 0]
    std_regret = regret.std(dim=-1).detach()[init_examples:, 0]

    # print(y_true.shape,y_true)
    # y_true = cum_res
    # mean_regret = y_true.mean(dim=-1).detach()[init_examples:, 0]
    # std_regret = y_true.std(dim=-1).detach()[init_examples:, 0]

    return mean_regret, std_regret

parser = argparse.ArgumentParser(
    description='Script to post_process results of BO experiments')
parser.add_argument('-e','--exps_dir', type=str, required=True,
    help='''Path to experiments directory where sub-directories contains
        results for different experiments to be compared''')
parser.add_argument('-s', '--per_std', type=float, default=0.1,
    help='Percentage of standard deviation to be plotted as errorbar')
parser.add_argument('-x', '--xlabel', type=str, required=True,
    help='Label for the x-axis of the plot')
parser.add_argument('-y', '--ylabel', type=str, required=True,
    help='Label for the y-axis of the plot')
args = parser.parse_args()

print(os.listdir(args.exps_dir))

exp_list = ['ZOBO','topk_max','Prabu_FOBO']
# i = 0
# plt.figure()
fig, axs = plt.subplots(3, 3,figsize = (12,12))

for exp in os.listdir(args.exps_dir):
    if exp != '.DS_Store':
        for experiments in exp_list:
            try:
                exp_path = os.path.join(args.exps_dir, exp) 
                exp_path = os.path.join(exp_path, experiments) 
                print(exp_path)
                config_file = os.path.join(exp_path, 'config.yaml')
                with open(config_file, 'r') as file:
                    config = yaml.safe_load(file)
                obj_fn = build_obj_func(config)

                y,X,y_true = combine_output(config, exp_path)
                # print('X:',X)
                # print('y:',y)

                mean_regret, std_regret = process_output(y,X,y_true, obj_fn,
                    config['init_examples'], mode=config['max/min'])
                x_vals = range(1, mean_regret.shape[0] + 1)
                # x_vals = range(1, 26)

                _,_,bars = axs[0][0].errorbar(x_vals, mean_regret, args.per_std*std_regret,
                    linestyle = 'solid', label=config['experiment_name'])

                [bar.set_alpha(0.2) for bar in bars]

            # true_vals = np.ones(mean_regret.shape[0])*obj_fn.true_opt_value
            # plt.plot(x_vals,true_vals,linestyle = 'dashed',color = 'black')
            except Exception as e:
                import traceback
                traceback.print_exc()

# # print(str(obj_fn.low))
# # plt.title(str(obj_fn).replace("()","")+':\nlow:'+str(obj_fn.low)+'\nhigh:'+str(obj_fn.high))
        plt.legend()
        axs[0][0].set(xlabel=args.xlabel,ylabel=args.ylabel)
        # plt.xlabel('Number of iterations')
        # plt.ylabel('y_true')
        axs[0][0].legend()
plt.savefig(os.path.join(args.exps_dir, 'Regret_plot.png'))
# plt.savefig(os.path.join(args.exps_dir, 'y_true.png'))
