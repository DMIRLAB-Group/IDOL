import copy
import datetime
import logging
import os
import random

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

plt.switch_backend('agg')


def setSeed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def adjust_learning_rate(optimizer, epoch, args, logger=None):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type0':
        return
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
        if epoch > 4:
            return
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if logger != None:
            logger.info('Updating learning rate to {}'.format(lr))
        else:
            print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.best_model_params = None

    def __call__(self, val_loss, model, path, is_saved=True):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, is_saved)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, is_saved)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path, is_save):
        if is_save:
            torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). ')
        self.best_model_params = copy.deepcopy(model.state_dict())
        self.val_loss_min = val_loss


def getLogger(name, path, time):
    logger = logging.getLogger(name + time)
    logger.setLevel('INFO')
    os.makedirs(path, exist_ok=True)
    fileHander = logging.FileHandler(filename=f'{path}/{time}.txt', mode='w')
    streamHander = logging.StreamHandler()
    format = logging.Formatter(fmt='%(message)s')
    streamHander.setFormatter(format)
    logger.addHandler(streamHander)
    logger.addHandler(fileHander)
    return logger


def create_path_gw(args):
    path = f"{args.checkpoints}/{args.model}/{args.root_path.rsplit('/')[-2]}_{args.data_path.replace('.csv', '')}"
    os.makedirs(path, exist_ok=True)
    time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger = getLogger(args.loss, f"{path}/{args.loss}_{args.seq_len}_{args.pred_len}", time)
    return path, logger


def create_path(args):
    path = f"{args.checkpoints}/{args.model}/{args.root_path.rsplit('/')[-2]}_{args.data_path.replace('.csv', '')}"
    os.makedirs(path, exist_ok=True)
    time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger = getLogger("", f"{path}/{args.seq_len}_{args.pred_len}", time)
    return path, logger


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def get_train_val_test_index(length, train_rate, val_rate):
    train_index = np.random.choice(range(length), int(length * train_rate), replace=False)

    temp_index = list(set(range(length)).difference(set(train_index)))
    val_index = np.random.choice(temp_index, int(length * val_rate), replace=False)
    test_index = list(set(temp_index).difference(set(val_index)))
    return train_index, val_index, test_index


def get_setting(args):
    setting = f"lr:{args.learning_rate};patience:{args.patience};train_epochs:{args.train_epochs};batch_size:{args.batch_size};lradj:{args.lradj};seed:{args.seed}"
    loss_name = args.loss
    if loss_name == "mse":

        add_setting = ""
    elif loss_name == "GWW":

        add_setting = f"beta:{args.beta},inner_N:{args.inner_N}"
    elif loss_name == "dilate":

        add_setting = f"dilate_weight:{args.dilate_alpha},gamma:{args.gamma}"
    elif loss_name == 'new_gw':

        add_setting = f" ;gw_lr:{args.lr},iteration:{args.iteration},w_weight:{args.w_weight},gw_weight:{args.gw_weight},c_weight:{args.c_weight},move_size:{args.move_sizes}"
    elif loss_name == 'gw_cc':

        add_setting = f" ; gw_lr:{args.lr},iteration_range:{args.iteration_range},w_rate_range:{args.w_rate_range}, gw_rate:{args.gw_rate_range},c_rate: {args.c_rate_range},  " \
                      f"move_size:{args.move_sizes},update_type:{args.gw_type},adj_weight_type:{args.adj_weight_type},feature_cat_type:{args.feature_cat_type}" \
                      f" cost_type:{args.cost_type}"
    elif loss_name == 'final_gw':
        add_setting = f" ; gw_lr:{args.lr},iteration_range:{args.iteration_range},w_rate_range:{args.w_rate_range}, gw_rate:{args.gw_rate_range},c_rate: {args.c_rate_range},  " \
                      f"move_size:{args.move_sizes}"
    return setting + add_setting


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)
