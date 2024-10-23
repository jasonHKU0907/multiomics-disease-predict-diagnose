
import numpy as np
import pandas as pd
import re
import os
import glob
from tqdm import tqdm
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV

group = 'Prevalent'
nb_cpus = 10
my_seed = 2024
fold_id_lst = [i for i in range(10)]

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

def pred_iter(mydf, omics_f_lst, my_params, tgt_name, my_seed, group, fold_id):
    train_idx = mydf['Split'].index[mydf['Split'] != fold_id]
    test_idx = mydf['Split'].index[mydf['Split'] == fold_id]
    X_train, y_train = mydf.iloc[train_idx][omics_f_lst], mydf.iloc[train_idx].target_y
    X_test, y_test = mydf.iloc[test_idx][omics_f_lst], mydf.iloc[test_idx].target_y
    my_lgb = LGBMClassifier(objective='binary', metric='auc', is_unbalance=True, seed=my_seed)
    my_lgb.set_params(**my_params)
    calibrate = CalibratedClassifierCV(my_lgb, method='isotonic', cv=5)
    calibrate.fit(X_train, y_train)
    y_pred_train_lst = calibrate.predict_proba(X_train)[:, 1].tolist()
    eid_train_lst = mydf.iloc[train_idx].eid.tolist()
    pred_train_df = pd.DataFrame({'eid': eid_train_lst, 'risk_score': y_pred_train_lst})
    pred_train_df = pd.merge(mydf[['eid', 'Split', 'target_y', 'BL2Target_yrs']], pred_train_df, how='inner', on=['eid'])
    pred_train_df.sort_values(by='eid', ascending=True, inplace=True)
    pred_train_df.to_csv(dpath + 'Results/' + group + \
                        '/MultiOmics/IntegratedOmics/DirectIntegration/S3_RiskScore/TrainData/SplitFold_wo_' + \
                        str(fold_id) + '/' + tgt_name + '.csv', index=False)
    y_pred_test_lst = calibrate.predict_proba(X_test)[:, 1].tolist()
    eid_test_lst = mydf.iloc[test_idx].eid.tolist()
    pred_test_df = pd.DataFrame({'eid': eid_test_lst, 'risk_score': y_pred_test_lst})
    return pred_test_df

tgt_name_df = pd.read_csv(dpath + 'Data/TargetData/DiseaseTable.csv')
tgt_name_df = tgt_name_df.loc[tgt_name_df[group + 'Analysis'] == 1]
tgt_name_lst = tgt_name_df.NAME.tolist()
tgt_name_lst.sort()

eid_df = pd.read_csv(dpath + 'Data/BloodData/BloodDataIndex.csv')
eid_df = eid_df.loc[eid_df.Test == 1]
nmr_df = pd.read_csv(dpath + 'Data/BloodData/MetabolomicData/MetabolomicData.csv')
pro_df = pd.read_csv(dpath + 'Data/BloodData/ProteomicData/ProteomicData.csv')
ser_df = pd.read_csv(dpath + 'Data/BloodData/SerumData/SerumData.csv')

omics_df_full = pd.merge(eid_df, nmr_df, how = 'inner', on = ['eid'])
omics_df_full = pd.merge(omics_df_full, pro_df, how = 'inner', on = ['eid'])
omics_df_full = pd.merge(omics_df_full, ser_df, how = 'inner', on = ['eid'])
#omics_f_gen = ['Pt_1e-06', 'Pt_5e-06', 'Pt_1e-05', 'Pt_5e-05', 'Pt_0.0001', 'Pt_0.0005',
#               'Pt_0.001', 'Pt_0.005', 'Pt_0.01', 'Pt_0.05', 'Pt_0.1', 'Pt_0.5', 'Pt_1']


for tgt_name in tqdm(tgt_name_lst):
    tgt_df = read_target(dpath, tgt_name, group)
    omics_df, omics_f_lst = get_omics_df(dpath, group, omics_df_full, tgt_name)
    gen_df = pd.read_csv(dpath + 'Data/BloodData/GenomicData/GenomicData/' + tgt_name + '.csv')
    omics_f_gen = gen_df.columns.tolist()[1:]
    omics_df = pd.merge(omics_df, gen_df, how = 'inner', on = ['eid'])
    omics_f_lst += omics_f_gen
    mydf = pd.merge(tgt_df, omics_df, how = 'inner', on = ['eid'])
    param_df = pd.read_csv(dpath + 'Results/' + group + '/MultiOmics/IntegratedOmics/DirectIntegration/S2_ParameterSelection/' + tgt_name + '.csv')
    my_params = dict(param_df.iloc[0, :6])
    my_params['n_estimators'], my_params['max_depth'], my_params['num_leaves'] = int(my_params['n_estimators']), int(my_params['max_depth']), int(my_params['num_leaves'])
    pred_df_lst = Parallel(n_jobs=nb_cpus)(delayed(pred_iter)(mydf, omics_f_lst, my_params, tgt_name, my_seed, group, fold_id) for fold_id in fold_id_lst)
    y_pred_test = pd.DataFrame()
    for pred_df in pred_df_lst:
        y_pred_test = pd.concat([y_pred_test, pred_df], axis=0)
    y_pred_test = pd.merge(mydf[['eid', 'Split', 'target_y', 'BL2Target_yrs']], y_pred_test, how='inner', on=['eid'])
    y_pred_test.sort_values(by='eid', ascending=True, inplace=True)
    y_pred_test.to_csv(dpath + 'Results/' + group + '/MultiOmics/IntegratedOmics/DirectIntegration/S3_RiskScore/TestData/' + tgt_name + '.csv', index=False)



