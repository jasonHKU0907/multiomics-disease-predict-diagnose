
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm
from lightgbm import LGBMClassifier
from joblib import Parallel, delayed

my_omics = 'Proteomic'
group = 'Incident'
nb_cpus = 10
my_seed = 2024
fold_id_lst = [i for i in range(10)]

dpath = '/home1/jiayou/Documents/Projects/BloodOmicsPred/'
#dpath = '/Volumes/JasonWork/Projects/BloodOmicsPred/'

my_params = {'n_estimators': 100,
             'max_depth': 15,
             'num_leaves': 10,
             'subsample': 0.7,
             'learning_rate': 0.01,
             'colsample_bytree': 0.7}

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


def normal_imp(mydict):
    mysum = sum(mydict.values())
    mykeys = mydict.keys()
    for key in mykeys:
        mydict[key] = mydict[key]/mysum
    return mydict


def model_train_iter(mydf, omics_f_lst, my_seed, my_params, fold_id):
    train_idx = mydf['Split'].index[mydf['Split'] != fold_id]
    X_train, y_train = mydf.iloc[train_idx][omics_f_lst], mydf.iloc[train_idx].target_y
    my_lgb = LGBMClassifier(objective='binary', metric='auc', is_unbalance=True, seed=my_seed)
    my_lgb.set_params(**my_params)
    my_lgb.fit(X_train, y_train)
    totalgain_imp = my_lgb.booster_.feature_importance(importance_type='gain')
    totalgain_imp = dict(zip(my_lgb.booster_.feature_name(), totalgain_imp.tolist()))
    return totalgain_imp


tgt_name_df = pd.read_csv(dpath + 'Data/TargetData/DiseaseTable.csv')
tgt_name_df = tgt_name_df.loc[tgt_name_df[group + 'Analysis'] == 1]
tgt_name_lst = tgt_name_df.NAME.tolist()
tgt_name_lst.sort()

omics_df = pd.read_csv(dpath + 'Data/BloodData/'+my_omics+'Data/'+my_omics+'Data.csv')
omics_f_lst = omics_df.columns.tolist()[1:]
eid_df = pd.read_csv(dpath + 'Data/BloodData/BloodDataIndex.csv')
eid_df = eid_df.loc[(eid_df[my_omics] == 1)&(eid_df.Test != 1)]
omics_df = pd.merge(omics_df, eid_df, how = 'inner', on = ['eid'])
f_df = pd.DataFrame({'Omics_feature': omics_f_lst})
f_df.sort_values(by='Omics_feature', ascending=True, inplace=True)

bad_tgt_lst = []

for tgt_name in tqdm(tgt_name_lst):
    tgt_df = read_target(dpath, tgt_name, group)
    mydf = pd.merge(tgt_df, omics_df, how='inner', on=['eid'])
    tg_imp_cv = Counter()
    try:
        tg_imp_lst = Parallel(n_jobs=nb_cpus)(delayed(model_train_iter)(mydf, omics_f_lst, my_seed, my_params, fold_id) for fold_id in fold_id_lst)
        for tg_imp in tg_imp_lst:
            tg_imp_cv += Counter(normal_imp(tg_imp))
        tg_imp_cv = normal_imp(tg_imp_cv)
        tg_imp_df = pd.DataFrame({'Omics_feature': list(tg_imp_cv.keys()), 'Importance': list(tg_imp_cv.values())})
        tg_imp_df = pd.merge(f_df, tg_imp_df, how='left', on=['Omics_feature'])
        tg_imp_df['Importance'].fillna(0, inplace=True)
        tg_imp_df.sort_values(by='Importance', ascending=False, inplace=True)
        tg_imp_df.to_csv(dpath + 'Results/' + group + '/SingleOmics/' + my_omics + '/S0_FeatureImportance/' + tgt_name + '.csv', index=False)
    except:
        bad_tgt_lst.append(tgt_name)

bad_tgt_df = pd.DataFrame(bad_tgt_lst)
bad_tgt_df.to_csv(dpath + 'Results/' + group + '/SingleOmics/' + my_omics + '/Bad_FI_Target.csv', index = False)


