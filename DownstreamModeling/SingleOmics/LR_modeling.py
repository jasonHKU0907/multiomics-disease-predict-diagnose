
import glob
import re
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import statsmodels.api as sm
from joblib import Parallel, delayed

group = 'Prevalent'
nb_cpus = 10
my_seed = 2024
dpath = '/home1/jiayou/Documents/Projects/BloodOmicsPred/'
#dpath = '/Volumes/JasonWork/Projects/BloodOmicsPred/'

omics_lst = ['Genomic', 'Metabolomic', 'Proteomic', 'Serum']

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

def get_cov_df(dpath, cov_df, eid_df, group, tgt_name):
    tgt_df = read_target(dpath, tgt_name, group)
    eid_train_df = eid_df.loc[eid_df.Test != 1]
    eid_test_df = eid_df.loc[eid_df.Test == 1]
    cov_train_df = pd.merge(cov_df, eid_train_df, how='inner', on=['eid'])
    cov_test_df = pd.merge(cov_df, eid_test_df, how='inner', on=['eid'])
    cov_traindf = pd.merge(cov_train_df, tgt_df, how='inner', on=['eid'])
    cov_testdf = pd.merge(cov_test_df, tgt_df, how='inner', on=['eid'])
    return (cov_traindf, cov_testdf)

def get_omics_df(dpath, cov_df, group, my_omics, tgt_name):
    tgt_omics_traindf = pd.read_csv(dpath + 'Results/' + group + '/SingleOmics/' + my_omics + '/S3_RiskScore/TrainData/' + tgt_name + '.csv')
    omics_traindf = pd.merge(tgt_omics_traindf, cov_df, how='inner', on=['eid'])
    omics_traindf['risk_score'] = (omics_traindf.risk_score-omics_traindf.risk_score.mean())/omics_traindf.risk_score.std()
    tgt_omics_testdf = pd.read_csv(dpath + 'Results/' + group + '/SingleOmics/' + my_omics + '/S3_RiskScore/TestData/' + tgt_name + '.csv')
    omics_testdf = pd.merge(tgt_omics_testdf, cov_df, how='inner', on=['eid'])
    omics_testdf['risk_score'] = (omics_testdf.risk_score-omics_traindf.risk_score.mean())/omics_traindf.risk_score.std()
    return (omics_traindf, omics_testdf)

def pred_logit(traindf, testdf, my_f_lst, col_name):
    train_y = traindf.target_y
    train_x = traindf[my_f_lst]
    test_x = testdf[my_f_lst]
    try:
        log_mod = sm.Logit(train_y, sm.add_constant(train_x)).fit()
    except:
        log_mod = sm.Logit(train_y, sm.add_constant(train_x)).fit(method = 'lbfgs', maxiter = 100)
    y_pred = log_mod.predict(sm.add_constant(test_x))
    y_pred_df = pd.DataFrame({'eid': testdf.eid, col_name + '_LogitRisk': y_pred})
    return y_pred_df

tgt_info_df = pd.read_csv(dpath + 'Data/TargetData/DiseaseTable.csv')
tgt_name_lst = tgt_info_df.loc[tgt_info_df[group + 'Analysis'] == 1].NAME.tolist()
tgt_name_lst.sort()

tgt_dir_lst0 = glob.glob(dpath + 'Results/' + group + '/MultiOmics/RiskScoreRefit/LogitRisk/SingleOmics/*.csv')
tgt_name_lst0 = [os.path.basename(tgt_dir)[:-4] for tgt_dir in tgt_dir_lst0]
tgt_name_lst = [tgt_name for tgt_name in tgt_name_lst if tgt_name not in tgt_name_lst0]


eid_df = pd.read_csv(dpath + 'Data/BloodData/BloodDataIndex.csv', usecols=['eid', 'Test', 'Split'])

cov_df = pd.read_csv(dpath + 'Data/Covariates/Covariates.csv')
cov_df['Race'].replace([1,2,3,4], [1, 0, 0, 0], inplace = True)

cov_fml = ['Age', 'Sex', 'Race', 'TDI', 'BMI']
cov_fml_sex = ['Age', 'Race', 'TDI', 'BMI']

def process(dpath, cov_df, eid_df, group, omics_lst, tgt_info_df, cov_fml, cov_fml_sex, tgt_name):
    sex_id = tgt_info_df.loc[tgt_info_df.NAME == tgt_name].SEX.iloc[0]
    my_fml_cov = [cov_fml_sex if (sex_id == 1) | (sex_id == 2) else cov_fml][0]
    my_fml_omics_cov = my_fml_cov + ['risk_score']
    cov_traindf, cov_testdf = get_cov_df(dpath, cov_df, eid_df, group, tgt_name)
    y_pred_cov = pred_Logit(cov_traindf, cov_testdf, my_fml_cov, 'Cov')
    y_pred = pd.merge(cov_testdf[['eid', 'Split', 'target_y', 'BL2Target_yrs']], y_pred_cov, how='inner', on=['eid'])
    for my_omics in omics_lst:
        omics_traindf, omics_testdf = get_omics_df(dpath, cov_df, group, my_omics, tgt_name)
        y_pred_omics = pred_logit(omics_traindf, omics_testdf, 'risk_score', my_omics)
        y_pred_omics_cov = pred_logit(omics_traindf, omics_testdf, my_fml_omics_cov, my_omics + '_Cov')
        y_pred = pd.merge(y_pred, y_pred_omics, how='inner', on=['eid'])
        y_pred = pd.merge(y_pred, y_pred_omics_cov, how='inner', on=['eid'])
    y_pred.to_csv(dpath + 'Results/' + group + '/MultiOmics/RiskScoreRefit/LogitRisk/SingleOmics/' + tgt_name + '.csv', index=False)
    return None

out = Parallel(n_jobs=nb_cpus)(delayed(process)(dpath, cov_df, eid_df, group, omics_lst, tgt_info_df, cov_fml,
                                                cov_fml_sex, tgt_name) for tgt_name in tgt_name_lst[:10])


