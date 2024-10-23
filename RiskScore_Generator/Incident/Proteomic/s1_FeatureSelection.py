
import numpy as np
import pandas as pd
import re
import os
import glob
from tqdm import tqdm
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

my_omics = 'Proteomic'
group = 'Incident'
nb_cpus = 10
my_seed = 2024
top_nb = 50
fold_id_lst = [i for i in range(10)]

dpath = '/home1/jiayou/Documents/Projects/BloodOmicsPred/'
#dpath = '/Volumes/JasonWork/Projects/BloodOmicsPred/'

my_params = {'n_estimators': 100,
             'max_depth': 15,
             'num_leaves': 10,
             'subsample': 0.7,
             'learning_rate': 0.01,
             'colsample_bytree': 0.7}


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


def get_top_f(sfs_f_df, top_nb):
    auc_lst = sfs_f_df.AUC_cv_MEAN.tolist()
    try:
        i = 0
        while ((auc_lst[i + 5] - auc_lst[i] > 0.001) | (auc_lst[i + 4] - auc_lst[i] > 0.001) |
               (auc_lst[i + 3] - auc_lst[i] > 0.001) | (auc_lst[i + 2] - auc_lst[i] > 0.001) |
               (auc_lst[i + 1] - auc_lst[i] > 0.001)):
            i += 1
        out_iter = i + 1
    except:
        out_iter = top_nb
    return out_iter

def model_train_iter(mydf, tmp_f_lst, my_seed, my_params, fold_id):
    train_idx = mydf['Split'].index[mydf['Split'] != fold_id]
    test_idx = mydf['Split'].index[mydf['Split'] == fold_id]
    X_train, y_train = mydf.iloc[train_idx][tmp_f_lst], mydf.iloc[train_idx].target_y
    X_test, y_test = mydf.iloc[test_idx][tmp_f_lst], mydf.iloc[test_idx].target_y
    my_lgb = LGBMClassifier(objective='binary', metric='auc', is_unbalance=True, seed=my_seed)
    my_lgb.set_params(**my_params)
    my_lgb.fit(X_train, y_train)
    y_pred_prob = my_lgb.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_prob)
    return auc_score

tgt_dir_lst = sort_nicely(glob.glob(dpath + 'Results/' + group + '/SingleOmics/' + my_omics + '/S0_FeatureImportance/*.csv'))
tgt_name_lst = [os.path.basename(tgt_dir)[:-4] for tgt_dir in tgt_dir_lst]

omics_df = pd.read_csv(dpath + 'Data/BloodData/'+my_omics+'Data/'+my_omics+'Data.csv')
omics_f_lst = omics_df.columns.tolist()[1:]
eid_df = pd.read_csv(dpath + 'Data/BloodData/BloodDataIndex.csv')
eid_df = eid_df.loc[(eid_df[my_omics] == 1)&(eid_df.Test != 1)]
omics_df = pd.merge(omics_df, eid_df, how = 'inner', on = ['eid'])
omics_dict = pd.read_csv(dpath + 'Data/BloodData/'+my_omics+'Data/'+my_omics+'Dict.csv', usecols = ['Omics_feature', 'Omics_name'])

bad_tgt_lst = []

for tgt_name in tqdm(tgt_name_lst):
    tgt_df = read_target(dpath, tgt_name, group)
    mydf = pd.merge(tgt_df, omics_df, how='inner', on=['eid'])
    imp_df = pd.read_csv(dpath + 'Results/' + group + '/SingleOmics/' + my_omics + '/S0_FeatureImportance/' + tgt_name + '.csv')
    sorted_f_lst = imp_df.Omics_feature.tolist()
    tmp_f_lst, sfs_out_lst = [], []
    try:
        for f in sorted_f_lst[:top_nb]:
            tmp_f_lst.append(f)
            auc_fold_lst = Parallel(n_jobs=nb_cpus)(delayed(model_train_iter)(mydf, tmp_f_lst, my_seed, my_params, fold_id) for fold_id in fold_id_lst)
            sfs_out_lst.append([f, np.round(np.mean(auc_fold_lst), 5), np.round(np.std(auc_fold_lst), 5)])
            print(f)
        sfs_f_df = pd.DataFrame(sfs_out_lst)
        sfs_f_df.columns = ['Omics_feature', 'AUC_cv_MEAN', 'AUC_cv_SD']
        nb_select = get_top_f(sfs_f_df, top_nb)
        sfs_f_df['Select'] = [1] * nb_select + [0] * (len(sfs_f_df) - nb_select)
        myout_df = pd.merge(sfs_f_df, omics_dict, how='left', on=['Omics_feature'])
        myout_df.to_csv(dpath + 'Results/' + group + '/SingleOmics/' + my_omics + '/S1_FeatureSelection/' + tgt_name + '.csv', index=False)
    except:
        bad_tgt_lst.append(tgt_name)


bad_tgt_df = pd.DataFrame(bad_tgt_lst)
bad_tgt_df.to_csv(dpath + 'Results/' + group + '/SingleOmics/' + my_omics + '/Bad_FS_Target.csv', index = False)

