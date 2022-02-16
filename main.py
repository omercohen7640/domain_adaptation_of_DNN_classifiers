
from __future__ import absolute_import, division, print_function

import argparse
import torch
import torch.optim as optim
from Utils.common import set_random_seed, create_result_dir, save_run_data, get_log_path, write_to_log
import optuna
import learn
import numpy as np

from config import *
import logging
torch.backends.cudnn.benchmark = True  # For speed improvement with models with fixed-length inputs


def objective(trial, trial_prm):
    b = trial.suggest_float("b", 0, 10e8)
    c = trial.suggest_float("c", 0.01, 10e6)
    trial_prm.b = b
    trial_prm.c = c
    return learn.k_fold_reverse_validation(trial_prm)


# -------------------------------------------------------------------------------------------
#  Set Parameters
# -------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()

# ----- Run Parameters ---------------------------------------------#

parser.add_argument('--run-name', type=str, help='Name of dir to save results in (if empty, name by time)',
                    default='')

#TODO: check seed in original DALC
parser.add_argument('--seed', type=int,  help='random seed',
                    default=1)

parser.add_argument('--test-batch-size', type=int,  help='input batch size for testing (reduce if memory is limited)',
                    default=32)

parser.add_argument('--k_fold_validation', type=int, help='input number of k in k-cross-validation',
                    default=5)
# ----- Task Parameters ---------------------------------------------#

parser.add_argument("--source_target", type=str, required=True, help="Defines the source and target (<source.target>).")

parser.add_argument('--limit_train_samples', type=int,
                    help='Upper limit for the number of training samples (0 = unlimited)',
                    default=0)

# ----- Algorithm Parameters ---------------------------------------------#
#TODO: check the parameters in original DALC
parser.add_argument('--batch-size', type=int, help='input batch size for training',
                    default=32)

parser.add_argument('--num-epochs', type=int, help='number of epochs to train',
                    default=10) # 300

parser.add_argument('--lr', type=float, help='learning rate (initial)',
                    default=1e-3)

parser.add_argument('--model_type', type=str, help="Standard or stochastic model",
                    default='Standard')  # 'Standard' / 'stochastic'

parser.add_argument('--model-name', type=str, help="Define model type (hypothesis class)'",
                    default='DAFcNet3')  # 'DAFcNet3' / ConvNet3 / 'FcNet3' / 'OmConvNet'

parser.add_argument('--loss_type', type=str, help="Define loss function type",
                    default='DA_PACBayes_loss')  # 'DA_PACBayes_loss' / 'DA_PACBayes_convex_loss'

parser.add_argument('--override_eps_std', type=float,
                    help='For debug: set the STD of epsilon variable for re-parametrization trick (default=1.0)',
                    default=1.0)

# -------------------------------------------------------------------------------------------

prm = parser.parse_args()
prm.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
set_random_seed(prm.seed)
create_result_dir(prm)


prm.log_var_init = {'mean': -10, 'std': 0.1} # The initial value for the log-var parameter (rho) of each weight

# Number of Monte-Carlo iterations (for re-parametrization trick):
prm.n_MC = 7
if prm.model_type == 'Standard':
    prm.n_mc = 1

# prm.use_randomness_schedeule = True # False / True
# prm.randomness_init_epoch = 0
# prm.randomness_full_epoch = 500000000

#  Define optimizer:
prm.optim_func, prm.optim_args = optim.Adam,  {'lr': prm.lr}
# prm.optim_func, prm.optim_args = optim.SGD, {'lr': prm.lr, 'momentum': 0.9}

# Learning rate decay schedule:
# prm.lr_schedule = {'decay_factor': 0.1, 'decay_epochs': [10, 30]}
prm.lr_schedule = {} # No decay

# Test type:
prm.test_type = 'MaxPosterior' # 'MaxPosterior' / 'MajorityVote'

# Generate task data set:
search_space = {
    "b": list(np.logspace(0, 8, num=20)),
    "c": list(np.logspace(-2, 6, num=20))
}
write_to_log('b_space = {}\nc_space= {}\n'.format(search_space['b'], search_space['c']), prm)
logger = logging.getLogger()

logger.setLevel(logging.INFO)  # Setup the root logger.
logger.addHandler(logging.FileHandler(get_log_path(prm), mode="a"))

objective_wrap = lambda trial: objective(trial, prm)
study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space), direction='minimize')
study.optimize(objective_wrap)

# save_run_data(prm, {'test_err': test_err})