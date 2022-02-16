

from __future__ import absolute_import, division, print_function

import timeit
from copy import deepcopy

import torch

from Models.stochastic_models import get_model
from Utils import common as cmn, data_gen
from Utils.Bayes_utils import run_test_Bayes, get_bayes_task_objective
from Utils.common import grad_step, correct_rate, get_loss_criterion, get_value
from Utils.PACBayes_DA_utils import l2_reg
from itertools import zip_longest
import numpy as np
from Utils.data_gen import DataLoadersGen


def run_learning(source_train_loader, target_train_loader, prm, verbose=1):

    # -------------------------------------------------------------------------------------------
    #  Setting-up
    # -------------------------------------------------------------------------------------------

    # Unpack parameters:
    optim_func, optim_args, lr_schedule = \
        prm.optim_func, prm.optim_args, prm.lr_schedule

    # Loss criterion
    loss_criterion = get_loss_criterion(prm.loss_type, prm.b, prm.c)

    n_batches = len(source_train_loader)

    # get model:
    model = get_model(prm)

    # post_model.set_eps_std(0.0) # DEBUG: turn off randomness

    #  Get optimizer:
    optimizer = optim_func(model.parameters(), **optim_args)

    # -------------------------------------------------------------------------------------------
    #  Training epoch  function
    # -------------------------------------------------------------------------------------------

    def run_train_epoch(i_epoch):

        # # Adjust randomness (eps_std)
        # if hasattr(prm, 'use_randomness_schedeule') and prm.use_randomness_schedeule:
        #     if i_epoch > prm.randomness_full_epoch:
        #         eps_std = 1.0
        #     elif i_epoch > prm.randomness_init_epoch:
        #         eps_std = (i_epoch - prm.randomness_init_epoch) / (prm.randomness_full_epoch - prm.randomness_init_epoch)
        #     else:
        #         eps_std = 0.0  #  turn off randomness
        #     post_model.set_eps_std(eps_std)

        # post_model.set_eps_std(0.00) # debug

        model.train()

        for batch_idx, (source_batch_data,target_batch_data) in enumerate(zip_longest(source_train_loader,target_train_loader)):

            # Monte-Carlo iterations:
            empirical_loss = 0
            l2_obj = 0
            n_MC = prm.n_MC

            # get batch:
            source_x, source_y = data_gen.get_batch_vars(source_batch_data, prm)
            target_x, target_y = data_gen.get_batch_vars(target_batch_data, prm)
            
            for i_MC in range(n_MC):           

                # calculate objective:
                source_outputs = model(source_x)
                target_outputs = model(target_x)
                empirical_loss_c = loss_criterion(source_y, source_outputs, target_outputs).sum()
                empirical_loss += (1 / source_outputs.shape[0]*n_MC) * empirical_loss_c
                l2_obj += l2_reg(model, prm) / (2 * n_MC * prm.b * prm.c)

            objective = empirical_loss + l2_obj

            # Take gradient step:
            grad_step(objective, optimizer, lr_schedule, prm.lr, i_epoch)

            # Print status:
            log_interval = 50
            if batch_idx % log_interval == 0:
                source_batch_acc = correct_rate(source_outputs, source_y)
                target_batch_acc = correct_rate(target_outputs, target_y)
                print(cmn.status_string(i_epoch, prm.num_epochs, batch_idx, n_batches, source_batch_acc, target_batch_acc, get_value(objective)))
    # -------------------------------------------------------------------------------------------
    #  Main Script
    # -------------------------------------------------------------------------------------------

    # Update Log file
    update_file = not verbose == 0
    cmn.write_to_log(cmn.get_model_string(model), prm, update_file=update_file)
    cmn.write_to_log('Total number of steps: {}'.format(n_batches * prm.num_epochs), prm, update_file=update_file)
    # cmn.write_to_log('Number of source training samples: {}\nNumber of target training samples: {}'.format(source_train_loader.dataset.X.shape[0],
    #                 source_train_loader.dataset.X.shape[0]), prm=prm, update_file=update_file)

    start_time = timeit.default_timer()

    # Run training epochs:
    for i_epoch in range(prm.num_epochs):
        run_train_epoch(i_epoch)

    # Test:
    # test_acc, test_loss = run_test_Bayes(model, source_valid_loader, target_valid_loader, loss_criterion, prm)

    stop_time = timeit.default_timer()
    # cmn.write_final_result(test_acc, stop_time - start_time, prm, result_name=prm.test_type)

    # test_err = 1 - test_acc
    return model


def classify(model, dataloader, prm):
    model.eval()
    y_hat_total = np.array([])
    y_total = np.array([])
    with torch.no_grad():
        for batch_data in dataloader:
            batch_x, batch_y = data_gen.get_batch_vars(batch_data, prm)
            y_hat = model(batch_x)
            y_hat[y_hat >=0] = 1
            y_hat[y_hat < 0] = -1
            y_hat_total = np.concatenate((y_hat_total, y_hat.cpu().numpy().squeeze()))
            y_total = np.concatenate((y_total, batch_y.cpu().numpy().squeeze()))
    error_rate = (y_total != y_hat_total).astype(int)
    error_rate = error_rate.sum()/error_rate.shape
    return y_hat_total, error_rate


def k_fold_reverse_validation(prm):
    data_loader_gen = DataLoadersGen(prm.source_target)
    error_rate_list = []
    cmn.write_to_log(f'b = {prm.b} c = {prm.c}', prm)
    for i in range(prm.k_fold_validation):
        dataloader_dict = data_loader_gen.get_k_loader(i, prm)
        prm.input_shape = dataloader_dict['n_features']
        print('Perform reverse cross validation')
        print('step 1: learn from sourceUtarget')
        model = run_learning(dataloader_dict['source_train'],dataloader_dict['target_train'], prm)
        print('step 2: use model to label target')
        y_target,_ = classify(model, dataloader_dict['target_train_no_shuffle'], prm)
        dataloader_dict['target_train'].dataset.Y = y_target
        print('step 3: learn from (target + estimated label)U(source without labels)')
        model = run_learning(dataloader_dict['target_train'],dataloader_dict['source_train'], prm)
        print('step 4: use model to re-label source  and evaluate model by comparing source labels')
        _, error_rate = classify(model, dataloader_dict['source_val'], prm)
        cmn.write_to_log(f'validation round number {i}, error rate: {error_rate}',prm)
        error_rate_list.append(error_rate)
    result = np.array(error_rate_list).mean()
    cmn.write_to_log(f'b = {prm.b} c = {prm.c} final result = {result}',prm)
    return result
