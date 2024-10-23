
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

def get_pred_testdata(traindf, testdf, omics_f_lst, my_params, my_seed):
    X_train, y_train = traindf[omics_f_lst], traindf.target_y
    X_test, y_test = testdf[omics_f_lst], testdf.target_y
    my_lgb = LGBMClassifier(objective='binary', metric='auc', is_unbalance=True, seed=my_seed)
    my_lgb.set_params(**my_params)
    calibrate = CalibratedClassifierCV(my_lgb, method='isotonic', cv=5)
    calibrate.fit(X_train, y_train)
    testdf['risk_score'] = calibrate.predict_proba(X_test)[:, 1].tolist()
    pred_df = testdf[['eid', 'Split', 'target_y', 'BL2Target_yrs', 'risk_score']]
    return pred_df

def pred_traindata_iter(traindf, omics_f_lst, my_params, my_seed, fold_id):
    train_idx = traindf['Split'].index[traindf['Split'] != fold_id]
    test_idx = traindf['Split'].index[traindf['Split'] == fold_id]
    X_train, y_train = traindf.iloc[train_idx][omics_f_lst], traindf.iloc[train_idx].target_y
    X_test, y_test = traindf.iloc[test_idx][omics_f_lst], traindf.iloc[test_idx].target_y
    my_lgb = LGBMClassifier(objective='binary', metric='auc', is_unbalance=True, seed=my_seed)
    my_lgb.set_params(**my_params)
    calibrate = CalibratedClassifierCV(my_lgb, method='isotonic', cv=5)
    calibrate.fit(X_train, y_train)
    y_pred_lst = calibrate.predict_proba(X_test)[:, 1].tolist()
    eid_lst = traindf.iloc[test_idx].eid.tolist()
    pred_fold_df = pd.DataFrame({'eid': eid_lst, 'risk_score': y_pred_lst})
    return pred_fold_df


tgt_name_df = pd.read_csv(dpath + 'Data/TargetData/DiseaseTable.csv')
tgt_name_lst = tgt_name_df.NAME.tolist()
tgt_name_lst.sort()
tgt_name_df_ukb_GWAS = tgt_name_df.loc[tgt_name_df.GWAS_accession.isna() == True]
tgt_name_lst_ukb_GWAS = tgt_name_df_ukb_GWAS.NAME.tolist()
tgt_name_lst_ukb_GWAS.sort()


tgt_dir_lst0 = glob.glob(dpath + 'Data/BloodData/'+my_omics+'Data/S3_RiskScore/TestData/*.csv')
tgt_name_lst0 = [os.path.basename(tgt_dir)[:-4] for tgt_dir in tgt_dir_lst0]
tgt_name_lst = [tgt_name for tgt_name in tgt_name_lst if tgt_name not in tgt_name_lst0]


eid_df_300k = pd.read_csv(dpath + 'Data/BloodData/BloodDataIndex_300K.csv')
eid_train_df_300k = eid_df_300k.loc[(eid_df_300k[my_omics] == 1)&(eid_df_300k.Test != 1)]
eid_df_full = pd.read_csv(dpath + 'Data/BloodData/BloodDataIndex.csv')
eid_train_df_full = eid_df_full.loc[(eid_df_full[my_omics] == 1)&(eid_df_full.Test != 1)]
eid_test_df = eid_df_full.loc[(eid_df_full[my_omics] == 1)&(eid_df_full.Test == 1)]

for tgt_name in tqdm(tgt_name_lst):
    tgt_df = pd.read_csv(dpath + 'Data/TargetData/TargetData/' + tgt_name + '.csv')
    omics_df = pd.read_csv(dpath + 'Data/BloodData/' + my_omics + 'Data/' + my_omics + 'Data/' + tgt_name + '.csv')
    omics_f_lst = omics_df.columns.tolist()[1:]
    omics_df = pd.merge(tgt_df, omics_df, how = 'inner', on = ['eid'])
    eid_train_df = [eid_train_df_300k if tgt_name in tgt_name_lst_ukb_GWAS else eid_train_df_full][0]
    traindf = pd.merge(eid_train_df, omics_df, how='inner', on=['eid'])
    testdf = pd.merge(eid_test_df, omics_df, how='inner', on=['eid'])
    param_df = pd.read_csv(dpath + 'Data/BloodData/'+my_omics+'Data/S2_ParameterSelection/' + tgt_name + '.csv')
    my_params = dict(param_df.iloc[0, :6])
    my_params['n_estimators'], my_params['max_depth'], my_params['num_leaves'] = int(my_params['n_estimators']), int(my_params['max_depth']), int(my_params['num_leaves'])
    try:
        pred_fold_df_lst = Parallel(n_jobs=nb_cpus)(delayed(pred_traindata_iter)(traindf, omics_f_lst, my_params, my_seed, fold_id) for fold_id in fold_id_lst)
        y_pred_train = pd.DataFrame()
        for pred_fold_df in pred_fold_df_lst:
            y_pred_train = pd.concat([y_pred_train, pred_fold_df], axis=0)
        y_pred_train = pd.merge(traindf[['eid', 'Split', 'target_y', 'BL2Target_yrs']], y_pred_train, how='inner', on=['eid'])
        y_pred_train.sort_values(by='eid', ascending=True, inplace=True)
        y_pred_train.to_csv(dpath + 'Data/BloodData/'+my_omics+'Data/S3_RiskScore/TrainData/' + tgt_name + '.csv',index=False)
        y_pred_test = get_pred_testdata(traindf, testdf, omics_f_lst, my_params, my_seed)
        y_pred_test.to_csv(dpath + 'Data/BloodData/'+my_omics+'Data/S3_RiskScore/TestData/' + tgt_name + '.csv',index=False)
    except:
        pass

