
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

my_omics = 'Serum'
group = 'Prevalent'
nb_cpus = 20
my_seed = 2024
nb_params = 99
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
        my_lgb = LGBMClassifier(objective='binary', metric='auc', is_unbalance=True, seed=my_seed)
        try:
            my_lgb.set_params(**my_params0)
            my_lgb.fit(X_train, y_train)
            y_pred_prob = my_lgb.predict_proba(X_test)[:, 1]
            auc_cv_lst.append(roc_auc_score(y_test, y_pred_prob))
        except:
            pass
    my_params0['AUC_cv_MEAN'] = np.round(np.mean(auc_cv_lst), 5)
    return my_params0

tgt_dir_lst = sort_nicely(glob.glob(dpath + 'Results/' + group + '/SingleOmics/' + my_omics + '/S1_FeatureSelection/*.csv'))
tgt_name_lst = [os.path.basename(tgt_dir)[:-4] for tgt_dir in tgt_dir_lst]

omics_df = pd.read_csv(dpath + 'Data/BloodData/'+my_omics+'Data/'+my_omics+'Data.csv')
eid_df = pd.read_csv(dpath + 'Data/BloodData/BloodDataIndex.csv')
eid_df = eid_df.loc[(eid_df[my_omics] == 1)&(eid_df.Test != 1)]
omics_df = pd.merge(omics_df, eid_df, how = 'inner', on = ['eid'])

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

bad_tgt_lst = []

for tgt_name in tqdm(tgt_name_lst):
    tgt_df = read_target(dpath, tgt_name, group)
    omics_f_df = pd.read_csv(dpath + 'Results/' + group + '/SingleOmics/' + my_omics + '/S1_FeatureSelection/' + tgt_name + '.csv')
    try:
        omics_f_df = omics_f_df.loc[omics_f_df.Select == 1]
        omics_f_lst = omics_f_df.Omics_feature.tolist()
        mydf = pd.merge(tgt_df, omics_df[['eid', 'Split'] + omics_f_lst], how='inner', on=['eid'])
        my_params_lst = Parallel(n_jobs=nb_cpus)(delayed(params_iter)(mydf, omics_f_lst, fold_id_lst, my_seed, my_params) for my_params in candidate_params_lst)
        params_df = pd.DataFrame(my_params_lst)
        params_df.sort_values(by='AUC_cv_MEAN', ascending=False, inplace=True)
        params_df.to_csv(dpath + 'Results/' + group + '/SingleOmics/' + my_omics + '/S2_ParameterSelection/' + tgt_name + '.csv', index=False)
    except:
        bad_tgt_lst.append(tgt_name)

bad_tgt_df = pd.DataFrame(bad_tgt_lst)
bad_tgt_df.to_csv(dpath + 'Results/' + group + '/SingleOmics/' + my_omics + '/Bad_PT_Target.csv', index = False)


