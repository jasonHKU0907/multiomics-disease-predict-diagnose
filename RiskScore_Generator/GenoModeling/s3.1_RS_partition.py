
import numpy as np
import pandas as pd
import re
import os
import glob
from tqdm import tqdm
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV

my_omics = 'Genomic'
group = 'Incident'
nb_cpus = 10
my_seed = 2024
fold_id_lst = [i for i in range(10)]

dpath = '/home1/jiayou/Documents/Projects/BloodOmicsPred/'
#dpath = '/Volumes/JasonWork/Projects/BloodOmicsPred/'

def sort_nicely(l):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c.replace("_","")) for c in re.split('([0-9]+)', key) ]
    l.sort( key=alphanum_key )
    return l

def read_target(dpath, tgt_name, group):
    tgt_df = pd.read_csv(dpath + 'Data/TargetData/TargetData/' + tgt_name + '.csv')
    if group == 'Incident':
        rm_bl_idx = tgt_df.index[tgt_df.BL2Target_yrs < 0]
    elif group == 'Prevalent':
        rm_bl_idx = tgt_df.index[(tgt_df.target_y == 1) & (tgt_df.BL2Target_yrs > 0)]
    else:
        rm_bl_idx = np.nan
    tgt_df.drop(rm_bl_idx, axis=0, inplace=True)
    tgt_df.reset_index(inplace=True, drop=True)
    return tgt_df

tgt_name_df = pd.read_csv(dpath + 'Data/TargetData/DiseaseTable.csv')
tgt_name_df = tgt_name_df.loc[tgt_name_df[group + 'Analysis'] == 1]
tgt_name_lst = tgt_name_df.NAME.tolist()


bad_tgt_lst = []
for tgt_name in tqdm(tgt_name_lst):
    tgt_df = read_target(dpath, tgt_name, group)
    tgt_df = tgt_df[['eid']]
    rs_df_train = pd.read_csv(dpath + 'Data/BloodData/'+my_omics+'Data/S3_RiskScore/TrainData/' + tgt_name + '.csv')
    rs_df_train = pd.merge(tgt_df, rs_df_train, how = 'inner', on = ['eid'])
    rs_df_train.to_csv(dpath + 'Results/' + group + '/SingleOmics/' + my_omics + '/S3_RiskScore/TrainData/' + tgt_name + '.csv',index=False)
    rs_df_test = pd.read_csv(dpath + 'Data/BloodData/'+my_omics+'Data/S3_RiskScore/TestData/' + tgt_name + '.csv')
    rs_df_test = pd.merge(tgt_df, rs_df_test, how = 'inner', on = ['eid'])
    rs_df_test.to_csv(dpath + 'Results/' + group + '/SingleOmics/' + my_omics + '/S3_RiskScore/TestData/' + tgt_name + '.csv',index=False)
    print(rs_df_test.shape)

