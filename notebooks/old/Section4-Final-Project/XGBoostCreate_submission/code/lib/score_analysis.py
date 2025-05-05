import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from numpy import *
from numpy import round
import pandas as pd
import pickle as pkl


def plot_scores(styled_logs,title=None,normalize=True, check=False):
    """
    """
    if check:
        check_styled_logs(styled_logs)
        
    plt.figure(figsize=(8, 6))

    patches=[]
    for st_log in styled_logs:
        style=st_log['style']
        log=st_log['log']
        
        #find normalization constants
        if normalize:
            all_pred=[]
            for i in range(len(log)):
                X=log[i]
                y_pred=X['y_pred']
                all_pred.append(y_pred)
            all_pred=concatenate(all_pred)
            _mean=mean(all_pred)
            _std=std(all_pred)
        else:
            _mean=0
            _std=1

        for i in range(len(log)):
            X=log[i]
            y_pred=X['y_pred']
            y_test=X['y_test']
            y_pred=(y_pred-_mean)/_std
            thresholds = sorted(unique(round(y_pred, 2)))
            error_pos, error_neg = get_error_values(y_pred, y_test, thresholds)
            neg_line, = plt.plot(thresholds,error_neg,style[0],lw=0.5)
            pos_line, = plt.plot(thresholds,error_pos,style[1],lw=0.5)
        patch = mpatches.Patch(color=st_log['label_color'], label=st_log['label'])
        patches.append(patch)
    plt.grid(linestyle='-', linewidth='0.5', color='gray')
    plt.xlabel('Score')
    plt.ylabel('Error %')

    plt.legend(handles=patches)
    if title:
        plt.title(title)
        return _mean,_std

def check_styled_logs(styled_logs):
    
    i=0
    for X in styled_logs:
        for k in ['log', 'style', 'label', 'label_color']:
            assert k in X
        print(f'checking item {i}, label={X["label"]}')
        i+=1
        assert type(X['label'])==str
        assert type(X['label_color'])==str
        style=X['style']
        assert type(style)==list
        for st in style:
            assert type(st)==str
        Log=X['log']

        for l in Log:
            y_pred=l['y_pred']
            y_test=l['y_test']

            assert y_pred.shape== y_test.shape
            assert y_pred.dtype==dtype('float32')
            assert y_test.dtype==dtype('int8')

def get_error_values(y_pred, y_test, thresholds):
    accuracy_1 = []
    accuracy_0 = []
    for thresh in thresholds:
        y_test_i = y_test[y_test == 1]
        y_pred_i = y_pred[y_test == 1]
        correct = sum(y_pred_i > thresh)
        accuracy_1.append(1.0 * correct / len(y_test_i))

        y_test_i = y_test[y_test == 0]
        y_pred_i = y_pred[y_test == 0]
        correct = sum(y_pred_i <= thresh)
        accuracy_0.append(1.0 * correct / len(y_test_i))
    
    error_1 = list(1 - array(accuracy_1))
    error_0 = list(1 - array(accuracy_0))
    return error_1, error_0
