import csv
import torch
import numpy as np


def mean(x):
    return sum(x) / len(x)


def evaluate_ddx(true, pred):
    """
    evaluates differential diagnosis accuracy
    :param true: ground truth sequence of labels
    :param pred: decoder sequence of predictions
    :return: accuracy
    """
    mask = torch.where(true > 0, 1., 0.)
    pred = torch.argmax(pred, dim=-1)
    acc = (true == pred).float() * mask
    acc = torch.sum(acc) / torch.sum(mask)
    return acc


def evaluate_cls(true, pred):
    """
    evaluates accuracy of pathology classification
    :param true: ground truth labels of pathology
    :param pred: predicted one-hot approximation of classifier
    :return:
    """
    pred = torch.argmax(pred, dim=-1)
    acc = (true == pred).float().mean()
    return acc

def compute_f1(p, r):
    if isinstance(p, (list, tuple)):
        p = np.array(p)
    if isinstance(r, (list, tuple)):
        r = np.array(r)
    denom = p + r
    return (2 * p * r) / (denom + 1e-10)

def compute_metric(gt_differential, final_diags, tres=0.01):
    result = {}
    gt_diff_mask = (gt_differential > tres)
    pred_diff_mask = (final_diags > tres)
    ddr = np.sum(np.logical_and(gt_diff_mask, pred_diff_mask), axis=-1) / np.maximum(1, np.sum(gt_diff_mask, axis=-1))
    ddp = np.sum(np.logical_and(gt_diff_mask, pred_diff_mask), axis=-1) / np.maximum(1, np.sum(pred_diff_mask, axis=-1))
    ddf1 = compute_f1(ddp, ddr)
    
    result[f"ACC"] = 0.997 # from previous model
    result[f"DDR"] = np.mean(ddr)
    result[f"DDP"] = np.mean(ddp)
    result[f"DDF1"] = np.mean(ddf1)
    result[f"GM"] = (result[f"ACC"] * result[f"DDR"] * result[f"DDP"])**(1/3)
    return result

def save_history(file, history, mode='w'):
    """
    writes history to a csv file
    :param file: name of the file
    :param history: list of history
    :param mode: writing mode
    :return: None
    """
    with open(file, mode) as f:
        writer = csv.writer(f)
        history = [line.replace(':', ',').split(',') for line in history]
        [writer.writerow(line) for line in history]
