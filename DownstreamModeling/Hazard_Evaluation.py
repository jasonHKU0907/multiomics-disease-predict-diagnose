
import glob
import re
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
import random
from lifelines.utils import concordance_index
from joblib import Parallel, delayed

group = 'Incident'
nb_cpus = 10
nb_iters = 1000
my_seed = 2024
dpath = '/home1/jiayou/Documents/Projects/BloodOmicsPred/'
#dpath = '/Volumes/JasonWork/Projects/BloodOmicsPred/'


def sort_nicely(l):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c.replace("_","")) for c in re.split('([0-9]+)', key) ]
    l.sort( key=alphanum_key )
    return l

def convert_output(result_df):
    result_df = result_df.T
    result_df['Median'] = result_df.median(axis=1)
    result_df['LBD'] = result_df.quantile(0.025, axis=1)
    result_df['UBD'] = result_df.quantile(0.975, axis=1)
    output_lst = []
    for i in range(len(result_df)):
        output_lst.append('{:.3f}'.format(result_df['Median'][i]) + ' [' + '{:.3f}'.format(result_df['LBD'][i]) + ' - ' + '{:.3f}'.format(result_df['UBD'][i]) + ']')
    result_df['C_index'] = output_lst
    return result_df['C_index']

def get_iter_output(mydf, gt_col, time_col, pred_f_lst, my_iter):
    tmp_random = np.random.RandomState(my_iter)
    bt_idx = tmp_random.choice(range(len(mydf)), size=len(mydf), replace=True)
    mydf_bt = mydf.copy()
    mydf_bt = mydf_bt.iloc[bt_idx, :]
    mydf_bt.reset_index(inplace=True, drop=True)
    c_idx_lst = [concordance_index(mydf_bt[time_col], -mydf_bt[pred_f], mydf_bt[gt_col]) for pred_f in pred_f_lst]
    return c_idx_lst

tgt_name_df = pd.read_csv(dpath + 'Data/TargetData/DiseaseTable.csv')
tgt_name_lst = tgt_name_df.loc[tgt_name_df[group + 'Analysis'] == 1].NAME.tolist()
tgt_name_lst.sort()
tgt_dict_df = pd.read_csv(dpath + 'Data/TargetData/DiseaseTable.csv', usecols = ['NAME', 'LONGNAME', 'Root', 'ICD_10'])

pred_f_lst = ['Cov_hazard',
              'Genomic_hazard', 'Genomic_Cov_hazard',
              'Metabolomic_hazard', 'Metabolomic_Cov_hazard',
              'Proteomic_hazard', 'Proteomic_Cov_hazard',
              'Serum_hazard', 'Serum_Cov_hazard',
              'Genomic_Metabolomic_hazard', 'Genomic_Metabolomic_Cov_hazard',
              'Genomic_Proteomic_hazard', 'Genomic_Proteomic_Cov_hazard',
              'Genomic_Serum_hazard', 'Genomic_Serum_Cov_hazard',
              'Metabolomic_Serum_hazard', 'Metabolomic_Serum_Cov_hazard',
              'Metabolomic_Proteomic_hazard', 'Metabolomic_Proteomic_Cov_hazard',
              'Proteomic_Serum_hazard', 'Proteomic_Serum_Cov_hazard',
              'MultiOmics_hazard', 'MultiOmics_Cov_hazard',
              'Integrated_hazard', 'Integrated_Cov_hazard']

for tgt_name in tqdm(tgt_name_lst[600:]):
    pred_lst = glob.glob(dpath + 'Results/' + group + '/MultiOmics/RiskScoreRefit/Hazard/*/'+tgt_name+'.csv')
    pred_lst0, pred_lst1, pred_lst2, pred_lst3 = pred_lst[0], pred_lst[1], pred_lst[2], pred_lst[3]
    pred_df0, pred_df1, pred_df2, pred_df3 = pd.read_csv(pred_lst0), pd.read_csv(pred_lst1), pd.read_csv(pred_lst2), pd.read_csv(pred_lst3)
    pred_df1.drop(['Split', 'target_y', 'BL2Target_yrs'], axis = 1, inplace=True)
    pred_df2.drop(['Split', 'target_y', 'BL2Target_yrs'], axis = 1, inplace=True)
    pred_df3.drop(['Split', 'target_y', 'BL2Target_yrs'], axis = 1, inplace=True)
    pred_df = pd.merge(pred_df0, pred_df1, how = 'inner', on = ['eid'])
    pred_df = pd.merge(pred_df, pred_df2, how = 'inner', on = ['eid'])
    pred_df = pd.merge(pred_df, pred_df3, how = 'inner', on = ['eid'])
    pred_df = pred_df[['eid', 'target_y', 'BL2Target_yrs']+pred_f_lst]
    iter_eval_lst = Parallel(n_jobs=nb_cpus)(delayed(get_iter_output)(pred_df, 'target_y', 'BL2Target_yrs', pred_f_lst, my_iter) for my_iter in range(nb_iters))
    iter_eval_df = pd.DataFrame(iter_eval_lst)
    iter_eval_df.columns = pred_f_lst
    eval_f_df = convert_output(iter_eval_df)
    eval_f_df.to_csv(dpath + 'Results/Incident/MultiOmics/RiskScoreRefit/Evaluation/'+tgt_name+'.csv', index = True)

