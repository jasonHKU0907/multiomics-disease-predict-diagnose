
import numpy as np
import pandas as pd
import re
import os
import random
import glob
from tqdm import tqdm
from itertools import product
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

group = 'Incident'
nb_cpus = 10
my_seed = 2024
nb_params = 49
fold_id_lst = [i for i in range(10)]
omics_lst = ['Genomic', 'Metabolomic', 'Proteomic', 'Serum']

dpath = '/home1/jiayou/Documents/Projects/BloodOmicsPred/'
#dpath = '/Volumes/JasonWork/Projects/BloodOmicsPred/'

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

def select_params_combo(my_dict, nb_items, my_seed):
    combo_list = [dict(zip(my_dict.keys(), v)) for v in product(*my_dict.values())]
    random.seed(my_seed)
    return random.sample(combo_list, nb_items)

def params_iter(mydf, omics_f_lst, fold_id_lst, my_seed, my_params):
    auc_cv_lst = []
    my_params0 = my_params.copy()
    for fold_id in fold_id_lst:
        train_idx = mydf['Split'].index[mydf['Split'] != fold_id]
        test_idx = mydf['Split'].index[mydf['Split'] == fold_id]
        X_train, y_train = mydf.iloc[train_idx][omics_f_lst], mydf.iloc[train_idx].target_y
        X_test, y_test = mydf.iloc[test_idx][omics_f_lst], mydf.iloc[test_idx].target_y
        my_lgb = LGBMClassifier(objective='binary', metric='auc', is_unbalance=True, seed=my_seed, verbose = -1)
        try:
            my_lgb.set_params(**my_params0)
            my_lgb.fit(X_train, y_train)
            y_pred_prob = my_lgb.predict_proba(X_test)[:, 1]
            auc_cv_lst.append(roc_auc_score(y_test, y_pred_prob))
        except:
            pass
    my_params0['AUC_cv_MEAN'] = np.round(np.mean(auc_cv_lst), 5)
    return my_params0

def get_omic_f(dpath, group, my_omics, tgt_name):
    omics_f_df = pd.read_csv(dpath + 'Results/' + group + '/SingleOmics/' + my_omics + '/S1_FeatureSelection/' + tgt_name + '.csv')
    omics_f_df = omics_f_df.loc[omics_f_df.Select == 1]
    omics_f_lst = omics_f_df.Omics_feature.tolist()
    return omics_f_lst

def get_omics_df(dpath, group, mydf, tgt_name):
    omics_f_lst1 = get_omic_f(dpath, group, 'Metabolomic', tgt_name)
    omics_f_lst2 = get_omic_f(dpath, group, 'Proteomic', tgt_name)
    omics_f_lst3 = get_omic_f(dpath, group, 'Serum', tgt_name)
    omics_f_lst = omics_f_lst1 + omics_f_lst2 + omics_f_lst3
    tgt_omics_df = mydf[['eid', 'Split'] + omics_f_lst]
    return tgt_omics_df, omics_f_lst


my_params = {'n_estimators': 500,
             'max_depth': 15,
             'num_leaves': 10,
             'subsample': 0.7,
             'learning_rate': 0.01,
             'colsample_bytree': 0.7}

params_dict = {'n_estimators': [100, 200, 300, 400, 500],
               'max_depth': np.linspace(5, 30, 6).astype('int32').tolist(),
               'num_leaves': np.linspace(5, 30, 6).astype('int32').tolist(),
               'subsample': np.linspace(0.6, 1, 9).tolist(),
               'learning_rate': [0.1, 0.05, 0.01, 0.001],
               'colsample_bytree': np.linspace(0.6, 1, 9).tolist()}


candidate_params_lst = select_params_combo(params_dict, nb_params, my_seed)
candidate_params_lst = [my_params] + candidate_params_lst

tgt_name_df = pd.read_csv(dpath + 'Data/TargetData/DiseaseTable.csv')
tgt_name_df = tgt_name_df.loc[tgt_name_df[group + 'Analysis'] == 1]
tgt_name_lst = tgt_name_df.NAME.tolist()
tgt_name_lst.sort()


tgt_dir_lst0 = glob.glob(dpath + 'Results/' + group + '/MultiOmics/IntegratedOmics/DirectIntegration/S2_ParameterSelection/*.csv')
tgt_name_lst0 = [os.path.basename(tgt_dir)[:-4] for tgt_dir in tgt_dir_lst0]
tgt_name_lst = [tgt_name for tgt_name in tgt_name_lst if tgt_name not in tgt_name_lst0]

eid_df = pd.read_csv(dpath + 'Data/BloodData/BloodDataIndex.csv')
eid_df = eid_df.loc[eid_df.Test == 1]
nmr_df = pd.read_csv(dpath + 'Data/BloodData/MetabolomicData/MetabolomicData.csv')
pro_df = pd.read_csv(dpath + 'Data/BloodData/ProteomicData/ProteomicData.csv')
ser_df = pd.read_csv(dpath + 'Data/BloodData/SerumData/SerumData.csv')

omics_df_full = pd.merge(eid_df, nmr_df, how = 'inner', on = ['eid'])
omics_df_full = pd.merge(omics_df_full, pro_df, how = 'inner', on = ['eid'])
omics_df_full = pd.merge(omics_df_full, ser_df, how = 'inner', on = ['eid'])


for tgt_name in tqdm(tgt_name_lst):
    try:
        tgt_df = read_target(dpath, tgt_name, group)
        omics_df, omics_f_lst = get_omics_df(dpath, group, omics_df_full, tgt_name)
        gen_df = pd.read_csv(dpath + 'Data/BloodData/GenomicData/GenomicData/' + tgt_name + '.csv')
        omics_df = pd.merge(omics_df, gen_df, how='inner', on=['eid'])
        omics_f_lst += gen_df.columns.tolist()[1:]
        mydf = pd.merge(tgt_df, omics_df, how='inner', on=['eid'])
        my_params_lst = Parallel(n_jobs=nb_cpus)(delayed(params_iter)(mydf, omics_f_lst, fold_id_lst, my_seed, my_params) for my_params in candidate_params_lst)
        params_df = pd.DataFrame(my_params_lst)
        params_df.sort_values(by='AUC_cv_MEAN', ascending=False, inplace=True)
        params_df.to_csv(dpath + 'Results/' + group + '/MultiOmics/IntegratedOmics/DirectIntegration/S2_ParameterSelection/' + tgt_name + '.csv', index=False)
    except:
        pass

