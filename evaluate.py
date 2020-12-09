#-*- coding:utf-8 -*- 

import os
import torch as t
import torch.optim as optim
from torch.utils import data
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
np.random.seed(1080)

def LogisticLoss(y_hat, y, lossAlpha=4):
    return ( (1-y)*y_hat + (lossAlpha*y+1)*( (-y_hat).clamp(min=0) + (1+ (-y_hat.abs()).exp() ).log() ) ).mean()

def precision_recall_accuracy(pred, target):
    pred = (pred>0.5).float()
    pred_pos = pred.squeeze().nonzero().squeeze()
    target_pos = target.squeeze().nonzero().squeeze()
    
    truenum = t.sum(pred == target).cpu().item()
    acc = truenum / target.shape[0]
    
    if pred_pos.numel()==0:
        return 0.0, 0.0, acc
    elif pred_pos.numel()==1:
        pred_set = set([pred_pos.cpu().tolist()])
        target_set = set(target_pos.reshape([-1]).cpu().numpy())
        precision = len(pred_set&target_set) / (len(pred_set)+0.01)
        recall = len(pred_set&target_set) / (len(target_set)+0.01)
    else:
        pred_set = set(pred_pos.cpu().numpy())
        target_set = set(target_pos.reshape([-1]).cpu().numpy())
        precision = len(pred_set&target_set) / (len(pred_set)+0.01)
        recall = len(pred_set&target_set) / (len(target_set)+0.01)
    
    return precision,recall,acc

def evaluate_model(model, dataloader):
    model.eval()
    acc_list = []
    loss_list = []
    precision_list = []
    recall_list = []
    with t.set_grad_enabled(False):
        for (rna_name, seq_matrix),dot_matrix in dataloader:
            pred = model(seq_matrix)
            
            precision,recall,acc = precision_recall_accuracy(t.sigmoid(pred), dot_matrix)
            precision_list.append(precision)
            recall_list.append(recall)
            acc_list.append(acc)

            loss = LogisticLoss(pred, dot_matrix)
            loss_list.append(loss.item())
    
    model.train()
    return np.mean(acc_list), np.mean(loss_list), np.mean(precision_list), np.mean(recall_list)

def plot_acc_loss(session_name, hist_acc, hist_loss, hist_preci, hist_recall):
    train_acc = [ d[0] for d in hist_acc ]
    val_acc = [ d[1] for d in hist_acc ]
    train_loss = [ d[0] for d in hist_loss ]
    val_loss = [ d[1] for d in hist_loss ]
    train_preci = [ d[0] for d in hist_preci ]
    val_preci = [ d[1] for d in hist_preci ]
    train_recall = [ d[0] for d in hist_recall ]
    val_recall = [ d[1] for d in hist_recall ]
    
    plt.plot(range(1, 1+len(train_acc)), train_acc, c='#e91e63', label="training acc")
    plt.plot(range(1, 1+len(val_acc)), val_acc, c='#4caf50', label="validation acc")
    plt.legend(); plt.xlabel("epoch"); plt.ylabel("acc"); plt.title(session_name)
    plt.savefig(session_name+"_acc.png")
    plt.close()
    
    plt.plot(range(1, 1+len(train_loss)), train_loss, c='#e91e63', label="training loss")
    plt.plot(range(1, 1+len(val_loss)), val_loss, c='#4caf50', label="validation loss")
    plt.legend(); plt.xlabel("epoch"); plt.ylabel("loss"); plt.title(session_name)
    plt.savefig(session_name+"_loss.png")
    plt.close()

    plt.plot(range(1, 1+len(train_preci)), train_preci, c='#e91e63', label="training precision")
    plt.plot(range(1, 1+len(val_preci)), val_preci, c='#4caf50', label="validation precision")
    plt.legend(); plt.xlabel("epoch"); plt.ylabel("precision"); plt.title(session_name)
    plt.savefig(session_name+"_precision.png")
    plt.close()

    plt.plot(range(1, 1+len(train_recall)), train_recall, c='#e91e63', label="training recall")
    plt.plot(range(1, 1+len(val_recall)), val_recall, c='#4caf50', label="validation recall")
    plt.legend(); plt.xlabel("epoch"); plt.ylabel("recall"); plt.title(session_name)
    plt.savefig(session_name+"recall.png")
    plt.close()

