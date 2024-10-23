
import numpy as np
import pandas as pd
import re
import os
import glob
from tqdm import tqdm
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV

my_omics = 'Metabolomic'
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

tgt_dir_lst = sort_nicely(glob.glob(dpath + 'Results/' + group + '/SingleOmics/' + my_omics + '/S2_ParameterSelection/*.csv'))
tgt_name_lst = [os.path.basename(tgt_dir)[:-4] for tgt_dir in tgt_dir_lst]

omics_df = pd.read_csv(dpath + 'Data/BloodData/'+my_omics+'Data/'+my_omics+'Data.csv')
omics_f_lst = omics_df.columns.tolist()[1:]
eid_df = pd.read_csv(dpath + 'Data/BloodData/BloodDataIndex.csv')
eid_train_df = eid_df.loc[(eid_df[my_omics] == 1)&(eid_df.Test != 1)]
eid_test_df = eid_df.loc[(eid_df[my_omics] == 1)&(eid_df.Test == 1)]
omics_train_df = pd.merge(omics_df, eid_train_df, how = 'inner', on = ['eid'])
omics_test_df = pd.merge(omics_df, eid_test_df, how = 'inner', on = ['eid'])

bad_tgt_lst = []
for tgt_name in tqdm(tgt_name_lst):
    tgt_df = read_target(dpath, tgt_name, group)
    try:
        omics_f_df = pd.read_csv(dpath + 'Results/' + group + '/SingleOmics/' + my_omics + '/S1_FeatureSelection/' + tgt_name + '.csv')
        omics_f_df = omics_f_df.loc[omics_f_df.Select == 1]
        omics_f_lst = omics_f_df.Omics_feature.tolist()
        traindf = pd.merge(tgt_df, omics_train_df[['eid', 'Split'] + omics_f_lst], how='inner', on=['eid'])
        testdf = pd.merge(tgt_df, omics_test_df[['eid', 'Split'] + omics_f_lst], how='inner', on=['eid'])
        param_df = pd.read_csv(dpath + 'Results/' + group + '/SingleOmics/' + my_omics + '/S2_ParameterSelection/' + tgt_name + '.csv')
        my_params = dict(param_df.iloc[0, :6])
        my_params['n_estimators'], my_params['max_depth'], my_params['num_leaves'] = int(my_params['n_estimators']), int(my_params['max_depth']), int(my_params['num_leaves'])
        pred_fold_df_lst = Parallel(n_jobs=nb_cpus)(delayed(pred_traindata_iter)(traindf, omics_f_lst, my_params, my_seed, fold_id) for fold_id in fold_id_lst)
        y_pred_train = pd.DataFrame()
        for pred_fold_df in pred_fold_df_lst:
            y_pred_train = pd.concat([y_pred_train, pred_fold_df], axis=0)
        y_pred_train = pd.merge(traindf[['eid', 'Split', 'target_y', 'BL2Target_yrs']], y_pred_train, how='inner', on=['eid'])
        y_pred_train.sort_values(by='eid', ascending=True, inplace=True)
        y_pred_train.to_csv(dpath + 'Results/' + group + '/SingleOmics/' + my_omics + '/S3_RiskScore/TrainData/' + tgt_name + '.csv',index=False)
        y_pred_test = get_pred_testdata(traindf, testdf, omics_f_lst, my_params, my_seed)
        y_pred_test.to_csv(dpath + 'Results/' + group + '/SingleOmics/' + my_omics + '/S3_RiskScore/TestData/' + tgt_name + '.csv',index=False)
    except:
        bad_tgt_lst.append(tgt_name)

bad_tgt_df = pd.DataFrame(bad_tgt_lst)
bad_tgt_df.to_csv(dpath + 'Results/' + group + '/SingleOmics/' + my_omics + '/Bad_RS_Target.csv', index=False)
