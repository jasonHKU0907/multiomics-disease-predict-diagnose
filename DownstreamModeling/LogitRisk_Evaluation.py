
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

group = 'Prevalent'
nb_cpus = 10
nb_iters = 1000
my_seed = 2024
#dpath = '/home1/jiayou/Documents/Projects/BloodOmicsPred/'
dpath = '/Volumes/JasonWork/Projects/BloodOmicsPred/'


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


def get_iter_output(mydf, gt_col, pred_f_lst, ct_f_dict, my_iter):
    tmp_random = np.random.RandomState(my_iter)
    bt_idx = tmp_random.choice(range(len(mydf)), size=len(mydf), replace=True)
    mydf_bt = mydf.copy()
    mydf_bt = mydf_bt.iloc[bt_idx, :]
    mydf_bt.reset_index(inplace=True, drop=True)
    eval_f_lst = []
    for pred_f in pred_f_lst:
        eval_f_lst.append(get_eval(mydf_bt[gt_col], mydf_bt[pred_f], ct_f_dict[pred_f]))
    return eval_f_lst

def get_iter_output(mydf, gt_col, pred_f_lst, ct_f_dict, my_iter):
    tmp_random = np.random.RandomState(my_iter)
    bt_idx = tmp_random.choice(range(len(mydf)), size=len(mydf), replace=True)
    mydf_bt = mydf.copy()
    mydf_bt = mydf_bt.iloc[bt_idx, :]
    mydf_bt.reset_index(inplace=True, drop=True)
    eval_df = pd.DataFrame()
    for pred_f in pred_f_lst:
        eval_df = pd.concat([eval_df, get_eval(mydf_bt[gt_col], mydf_bt[pred_f], ct_f_dict[pred_f])], axis = 0)
    return eval_df

tgt_name_df = pd.read_csv(dpath + 'Data/TargetData/DiseaseTable.csv')
tgt_name_lst = tgt_name_df.loc[tgt_name_df[group + 'Analysis'] == 1].NAME.tolist()
tgt_name_lst.sort()
tgt_dict_df = pd.read_csv(dpath + 'Data/TargetData/DiseaseTable.csv', usecols = ['NAME', 'LONGNAME', 'Root', 'ICD_10'])

pred_f_lst = ['Cov_LogitRisk',
              'Genomic_LogitRisk', 'Genomic_Cov_LogitRisk',
              'Metabolomic_LogitRisk', 'Metabolomic_Cov_LogitRisk',
              'Proteomic_LogitRisk', 'Proteomic_Cov_LogitRisk',
              'Serum_LogitRisk', 'Serum_Cov_LogitRisk',
              'Genomic_Metabolomic_LogitRisk', 'Genomic_Metabolomic_Cov_LogitRisk',
              'Genomic_Proteomic_LogitRisk', 'Genomic_Proteomic_Cov_LogitRisk',
              'Genomic_Serum_LogitRisk', 'Genomic_Serum_Cov_LogitRisk',
              'Metabolomic_Serum_LogitRisk', 'Metabolomic_Serum_Cov_LogitRisk',
              'Metabolomic_Proteomic_LogitRisk', 'Metabolomic_Proteomic_Cov_LogitRisk',
              'Proteomic_Serum_LogitRisk', 'Proteomic_Serum_Cov_LogitRisk',
              'MultiOmics_LogitRisk', 'MultiOmics_Cov_LogitRisk',
              'Integrated_LogitRisk', 'Integrated_Cov_LogitRisk']

for tgt_name in tqdm(tgt_name_lst):
    pred_lst = glob.glob(dpath + 'Results/' + group + '/MultiOmics/RiskScoreRefit/LogitRisk/*/'+tgt_name+'.csv')
    pred_lst0, pred_lst1, pred_lst2, pred_lst3 = pred_lst[0], pred_lst[1], pred_lst[2], pred_lst[3]
    pred_df0, pred_df1, pred_df2, pred_df3 = pd.read_csv(pred_lst0), pd.read_csv(pred_lst1), pd.read_csv(pred_lst2), pd.read_csv(pred_lst3)
    pred_df1.drop(['Split', 'target_y', 'BL2Target_yrs'], axis = 1, inplace=True)
    pred_df2.drop(['Split', 'target_y', 'BL2Target_yrs'], axis = 1, inplace=True)
    pred_df3.drop(['Split', 'target_y', 'BL2Target_yrs'], axis = 1, inplace=True)
    pred_df = pd.merge(pred_df0, pred_df1, how = 'inner', on = ['eid'])
    pred_df = pd.merge(pred_df, pred_df2, how = 'inner', on = ['eid'])
    pred_df = pd.merge(pred_df, pred_df3, how = 'inner', on = ['eid'])
    pred_df = pred_df[['eid', 'Split', 'target_y', 'BL2Target_yrs']+pred_f_lst]
    opt_ct_lst = [Find_Optimal_Cutoff(pred_df.target_y, pred_df[pred_f])[0] for pred_f in pred_f_lst]
    ct_f_dict = dict(zip(pred_f_lst, opt_ct_lst))
    iter_eval_df_lst = Parallel(n_jobs=nb_cpus)(delayed(get_iter_output)(pred_df, 'target_y', pred_f_lst, ct_f_dict, my_iter) for my_iter in range(nb_iters))
    res_eval_df, i = pd.DataFrame(), 0
    for pred_f in pred_f_lst:
        eval_f_df = pd.DataFrame()
        for iter_eval_df in iter_eval_df_lst:
            eval_f_df = pd.concat([eval_f_df, iter_eval_df.iloc[i,:]], axis=1)
        eval_f_df = convert_output(eval_f_df.T)
        res_eval_df = pd.concat([res_eval_df, eval_f_df], axis=0)
        i += 1
    res_eval_df.index = pred_f_lst
    res_eval_df.to_csv(dpath + 'Results/Prevalent/MultiOmics/RiskScoreRefit/Evaluation/'+tgt_name+'.csv', index = True)


