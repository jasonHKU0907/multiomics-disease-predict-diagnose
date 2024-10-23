

import glob
import re
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
import random
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.metrics import roc_curve
from joblib import Parallel, delayed

my_omics = 'Metabolomic'
group = 'Prevalent'
nb_cpus = 20
nb_iters = 1000
my_seed = 2024

dpath = '/home1/jiayou/Documents/Projects/BloodOmicsPred/'
#dpath = '/Volumes/JasonWork/Projects/BloodOmicsPred/'


def sort_nicely(l):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c.replace("_","")) for c in re.split('([0-9]+)', key) ]
    l.sort( key=alphanum_key )
    return l


def threshold(array, cutoff):
    array1 = array.copy()
    array1[array1 < cutoff] = 0
    array1[array1 >= cutoff] = 1
    return array1


def Find_Optimal_Cutoff(target, predicted):
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]
    return list(roc_t['threshold'])


def get_eval(y_test, pred_prob, cutoff):
    pred_binary = threshold(pred_prob, cutoff)
    tn, fp, fn, tp = confusion_matrix(y_test, pred_binary).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    prec = tp / (tp + fp)
    Youden = sens + spec - 1
    f1 = 2 * prec * sens / (prec + sens)
    auc = roc_auc_score(y_test, pred_prob)
    evaluations = np.round((auc, acc, sens, spec, prec, Youden, f1), 5)
    evaluations = pd.DataFrame(evaluations).T
    evaluations.columns = ['AUC', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'Youden-index', 'F1-score']
    return evaluations

def convert_output(result_df):
    result_df = result_df.T
    result_df['Median'] = result_df.median(axis=1)
    result_df['LBD'] = result_df.quantile(0.025, axis=1)
    result_df['UBD'] = result_df.quantile(0.975, axis=1)
    output_lst = []
    for i in range(7):
        output_lst.append('{:.3f}'.format(result_df['Median'][i]) + ' [' +
                          '{:.3f}'.format(result_df['LBD'][i]) + ' - ' +
                          '{:.3f}'.format(result_df['UBD'][i]) + ']')
    result_df['output'] = output_lst
    series = result_df['output']
    return series.to_frame().T

def get_iter_output(mydf, gt_col, y_pred_col, opt_ct, my_iter):
    tmp_random = np.random.RandomState(my_iter)
    bt_idx = tmp_random.choice(range(len(mydf)), size=len(mydf), replace=True)
    mydf_bt = mydf.copy()
    mydf_bt = mydf_bt.iloc[bt_idx, :]
    mydf_bt.reset_index(inplace=True, drop=True)
    y_test_bt = mydf_bt[gt_col]
    eval_iter = get_eval(y_test_bt, mydf_bt[y_pred_col], opt_ct)
    return eval_iter


tgt_dir_lst = sort_nicely(glob.glob(dpath + 'Results/' + group + '/SingleOmics/' + my_omics + '/S3_RiskScore/TestData/*.csv'))
tgt_name_lst = [os.path.basename(tgt_dir)[:-4] for tgt_dir in tgt_dir_lst]

for tgt_name in tqdm(tgt_name_lst):
    tgt_pred_df = pd.read_csv(dpath + 'Results/' + group + '/SingleOmics/' + my_omics + '/S3_RiskScore/TestData/'+tgt_name+'.csv')
    opt_ct = Find_Optimal_Cutoff(tgt_pred_df.target_y, tgt_pred_df.risk_score)[0]
    iter_eval_lst = Parallel(n_jobs=nb_cpus)(delayed(get_iter_output)(tgt_pred_df, 'target_y', 'risk_score', opt_ct, my_iter) for my_iter in range(nb_iters))
    eval_df = pd.DataFrame()
    for iter_eval in iter_eval_lst:
        eval_df = pd.concat([eval_df, iter_eval], axis=0)
    res_eval = convert_output(eval_df)
    res_eval.to_csv(dpath + 'Results/' + group + '/SingleOmics/' + my_omics + '/S4_Evaluation/'+tgt_name+'.csv', index=True)

