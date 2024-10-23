
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

my_omics = 'Genomic'
nb_cpus = 10
my_seed = 2024
nb_params = 49
fold_id_lst = [i for i in range(10)]

dpath = '/home1/jiayou/Documents/Projects/BloodOmicsPred/'
#dpath = '/Volumes/JasonWork/Projects/BloodOmicsPred/'

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
        my_lgb = LGBMClassifier(objective='binary', metric='auc', is_unbalance=True, verbose = -1, seed=my_seed)
        try:
            my_lgb.set_params(**my_params0)
            my_lgb.fit(X_train, y_train)
            y_pred_prob = my_lgb.predict_proba(X_test)[:, 1]
            auc_cv_lst.append(roc_auc_score(y_test, y_pred_prob))
        except:
            pass
    my_params0['AUC_cv_MEAN'] = np.round(np.mean(auc_cv_lst), 5)
    return my_params0

tgt_name_df = pd.read_csv(dpath + 'Data/TargetData/DiseaseTable.csv')
tgt_name_lst = tgt_name_df.NAME.tolist()
tgt_name_lst.sort()
tgt_name_df_ukb_GWAS = tgt_name_df.loc[tgt_name_df.GWAS_access == 'UKB']
tgt_name_lst_ukb_GWAS = tgt_name_df_ukb_GWAS.NAME.tolist()
tgt_name_lst_ukb_GWAS.sort()

tgt_dir_lst0 = glob.glob(dpath + 'Data/BloodData/'+my_omics+'Data/S2_ParameterSelection/*.csv')
tgt_name_lst0 = [os.path.basename(tgt_dir)[:-4] for tgt_dir in tgt_dir_lst0]
tgt_name_lst = [tgt_name for tgt_name in tgt_name_lst if tgt_name not in tgt_name_lst0]


eid_df_300k = pd.read_csv(dpath + 'Data/BloodData/BloodDataIndex_300K.csv')
eid_df_300k = eid_df_300k.loc[(eid_df_300k[my_omics] == 1)&(eid_df_300k.Test != 1)]
eid_df_full = pd.read_csv(dpath + 'Data/BloodData/BloodDataIndex.csv')
eid_df_full = eid_df_full.loc[(eid_df_full[my_omics] == 1)&(eid_df_full.Test != 1)]

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
#omics_f_lst = ['Pt_1e-06', 'Pt_5e-06', 'Pt_1e-05', 'Pt_5e-05', 'Pt_0.0001', 'Pt_0.0005',
#               'Pt_0.001', 'Pt_0.005', 'Pt_0.01', 'Pt_0.05', 'Pt_0.1', 'Pt_0.5', 'Pt_1']


for tgt_name in tqdm(tgt_name_lst):
    try:
        tgt_df = pd.read_csv(dpath + 'Data/TargetData/TargetData/' + tgt_name + '.csv')
        omics_df = pd.read_csv(dpath + 'Data/BloodData/'+my_omics+'Data/'+my_omics+'Data/'+tgt_name+'.csv')
        omics_f_lst = omics_df.columns.tolist()[1:]
        eid_df = [eid_df_300k if tgt_name in tgt_name_lst_ukb_GWAS else eid_df_full][0]
        omics_df = pd.merge(omics_df, eid_df, how='inner', on=['eid'])
        mydf = pd.merge(tgt_df, omics_df[['eid', 'Split'] + omics_f_lst], how='inner', on=['eid'])
        my_params_lst = Parallel(n_jobs=nb_cpus)(delayed(params_iter)(mydf, omics_f_lst, fold_id_lst, my_seed, my_params) for my_params in candidate_params_lst)
        params_df = pd.DataFrame(my_params_lst)
        params_df.sort_values(by='AUC_cv_MEAN', ascending=False, inplace=True)
        params_df.to_csv(dpath + 'Data/BloodData/'+my_omics+'Data/S2_ParameterSelection/' + tgt_name + '.csv', index=False)
    except:
        pass

