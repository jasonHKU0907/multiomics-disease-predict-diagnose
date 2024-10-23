
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
fold_id_lst = [i for i in range(10)]
omics_lst = ['Genomic', 'Metabolomic', 'Proteomic', 'Serum']
omics0, omics1, omics2, omics3 = omics_lst[0], omics_lst[1], omics_lst[2], omics_lst[3]

def get_omics_df(dpath, group, my_omics, tgt_name):
    tgt_omics_testdf = pd.read_csv(dpath + 'Results/' + group + '/SingleOmics/' + my_omics + '/S3_RiskScore/TestData/' + tgt_name + '.csv')
    tgt_omics_testdf['risk_score'] = (tgt_omics_testdf.risk_score-tgt_omics_testdf.risk_score.mean())/tgt_omics_testdf.risk_score.std()
    tgt_omics_testdf.rename(columns = {'risk_score': my_omics}, inplace = True)
    return tgt_omics_testdf

def pred_logit_cv(mydf, fold_id_lst, my_f_lst, col_name):
    y_out_df = pd.DataFrame()
    for fold_id in fold_id_lst:
        train_idx, test_idx = mydf['Split'].index[mydf['Split'] != fold_id], mydf['Split'].index[mydf['Split'] == fold_id]
        traindf, testdf = mydf.iloc[train_idx], mydf.iloc[test_idx]
        train_y = traindf.target_y
        train_x, test_x = traindf[my_f_lst], testdf[my_f_lst]
        try:
            log_mod = sm.Logit(train_y, sm.add_constant(train_x)).fit()
        except:
            log_mod = sm.Logit(train_y, sm.add_constant(train_x)).fit(method='lbfgs')
        y_pred = log_mod.predict(sm.add_constant(test_x))
        y_pred_df = pd.DataFrame({'eid': testdf.eid, col_name + '_LogitRisk': y_pred})
        y_out_df = pd.concat([y_out_df, y_pred_df], axis=0)
    y_out_df.sort_values(by = 'eid', ascending=True, inplace = True)
    return y_out_df

tgt_info_df = pd.read_csv(dpath + 'Data/TargetData/DiseaseTable.csv')
tgt_name_lst = tgt_info_df.loc[tgt_info_df[group + 'Analysis'] == 1].NAME.tolist()
tgt_name_lst.sort()

tgt_dir_lst0 = glob.glob(dpath + 'Results/' + group + '/MultiOmics/RiskScoreRefit/LogitRisk/MultiOmics/*.csv')
tgt_name_lst0 = [os.path.basename(tgt_dir)[:-4] for tgt_dir in tgt_dir_lst0]
tgt_name_lst = [tgt_name for tgt_name in tgt_name_lst if tgt_name not in tgt_name_lst0]

cov_df = pd.read_csv(dpath + 'Data/Covariates/Covariates.csv')
cov_df['Race'].replace([1,2,3,4], [1, 0, 0, 0], inplace = True)

cov_fml = ['Age', 'Sex', 'Race', 'TDI', 'BMI']
cov_fml_sex = ['Age', 'Race', 'TDI', 'BMI']
my_fml_omics = ['Genomic', 'Metabolomic', 'Proteomic', 'Serum']

def process(dpath, cov_df, group, tgt_info_df, my_fml_omics, cov_fml_sex, cov_fml, omics0, omics1, omics2, omics3, tgt_name):
    sex_id = tgt_info_df.loc[tgt_info_df.NAME == tgt_name].SEX.iloc[0]
    my_fml_omics_cov = [my_fml_omics + cov_fml_sex if (sex_id == 1) | (sex_id == 2) else my_fml_omics + cov_fml][0]
    omics_df0 = get_omics_df(dpath, group, omics0, tgt_name)
    omics_df1 = get_omics_df(dpath, group, omics1, tgt_name)
    omics_df2 = get_omics_df(dpath, group, omics2, tgt_name)
    omics_df3 = get_omics_df(dpath, group, omics3, tgt_name)
    omics_df = pd.merge(cov_df, omics_df0, how='inner', on=['eid'])
    omics_df = pd.merge(omics_df, omics_df1[['eid', omics1]], how='inner', on=['eid'])
    omics_df = pd.merge(omics_df, omics_df2[['eid', omics2]], how='inner', on=['eid'])
    omics_df = pd.merge(omics_df, omics_df3[['eid', omics3]], how='inner', on=['eid'])
    y_pred_omics = pred_logit_cv(omics_df, fold_id_lst, my_fml_omics, 'MultiOmics')
    y_pred_omics_cov = pred_logit_cv(omics_df, fold_id_lst, my_fml_omics_cov, 'MultiOmics_Cov')
    y_pred = pd.merge(omics_df[['eid', 'Split', 'target_y', 'BL2Target_yrs']], y_pred_omics, how='inner', on=['eid'])
    y_pred = pd.merge(y_pred, y_pred_omics_cov, how='inner', on=['eid'])
    y_pred.to_csv(dpath + 'Results/' + group + '/MultiOmics/RiskScoreRefit/LogitRisk/MultiOmics/' + tgt_name + '.csv', index=False)
    return None

out = Parallel(n_jobs=nb_cpus)(delayed(process)(dpath, cov_df, group, tgt_info_df, my_fml_omics, cov_fml_sex, cov_fml,
                                                omics0, omics1, omics2, omics3, tgt_name) for tgt_name in tgt_name_lst)



