
import glob
import re
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from lifelines.utils import concordance_index
from lifelines import CoxPHFitter
from joblib import Parallel, delayed

group = 'Incident'
nb_cpus = 10
my_seed = 2024
dpath = '/home1/jiayou/Documents/Projects/BloodOmicsPred/'
#dpath = '/Volumes/JasonWork/Projects/BloodOmicsPred/'
fold_id_lst = [i for i in range(10)]
pair_omics_lst = [['Genomic', 'Metabolomic'], ['Genomic', 'Proteomic'], ['Genomic', 'Serum'],
                  ['Metabolomic', 'Serum'], ['Proteomic', 'Serum']]

def get_omics_df(dpath, cov_df, group, my_omics, tgt_name):
    tgt_omics_traindf = pd.read_csv(dpath + 'Results/' + group + '/SingleOmics/' + my_omics + '/S3_RiskScore/TrainData/' + tgt_name + '.csv')
    omics_traindf = pd.merge(tgt_omics_traindf, cov_df, how='inner', on=['eid'])
    omics_traindf['risk_score'] = (omics_traindf.risk_score-omics_traindf.risk_score.mean())/omics_traindf.risk_score.std()
    tgt_omics_testdf = pd.read_csv(dpath + 'Results/' + group + '/SingleOmics/' + my_omics + '/S3_RiskScore/TestData/' + tgt_name + '.csv')
    omics_testdf = pd.merge(tgt_omics_testdf, cov_df, how='inner', on=['eid'])
    omics_testdf['risk_score'] = (omics_testdf.risk_score-omics_traindf.risk_score.mean())/omics_traindf.risk_score.std()
    omics_traindf.rename(columns={'risk_score': my_omics}, inplace=True)
    omics_testdf.rename(columns={'risk_score': my_omics}, inplace=True)
    return (omics_traindf, omics_testdf)


def pred_hazard(traindf, testdf, formula, col_name):
    try:
        cph = CoxPHFitter()
        cph.fit(traindf, duration_col='BL2Target_yrs', event_col='target_y', formula=formula)
    except:
        try:
           cph = CoxPHFitter(penalizer=1e-5)
           cph.fit(traindf, duration_col='BL2Target_yrs', event_col='target_y', formula=formula)
        except:
            try:
                cph = CoxPHFitter(penalizer=0.001)
                cph.fit(traindf, duration_col='BL2Target_yrs', event_col='target_y', formula=formula)
            except:
                try:
                    cph = CoxPHFitter(penalizer=0.01)
                    cph.fit(traindf, duration_col='BL2Target_yrs', event_col='target_y', formula=formula)
                except:
                    cph = CoxPHFitter(penalizer=0.1)
                    cph.fit(traindf, duration_col='BL2Target_yrs', event_col='target_y', formula=formula)
    y_pred = cph.predict_partial_hazard(testdf)
    y_pred_df = pd.DataFrame({'eid': testdf.eid, col_name+'_hazard': y_pred})
    return y_pred_df


def pred_hazard_cv(mydf, fold_id_lst, formula, col_name):
    y_out_df = pd.DataFrame()
    for fold_id in fold_id_lst:
        train_idx, test_idx = mydf['Split'].index[mydf['Split'] != fold_id], mydf['Split'].index[mydf['Split'] == fold_id]
        traindf, testdf = mydf.iloc[train_idx], mydf.iloc[test_idx]
        try:
            cph = CoxPHFitter()
            cph.fit(traindf, duration_col='BL2Target_yrs', event_col='target_y', formula=formula)
        except:
            try:
                cph = CoxPHFitter(penalizer=1e-5)
                cph.fit(traindf, duration_col='BL2Target_yrs', event_col='target_y', formula=formula)
            except:
                try:
                    cph = CoxPHFitter(penalizer=0.001)
                    cph.fit(traindf, duration_col='BL2Target_yrs', event_col='target_y', formula=formula)
                except:
                    try:
                        cph = CoxPHFitter(penalizer=0.01)
                        cph.fit(traindf, duration_col='BL2Target_yrs', event_col='target_y', formula=formula)
                    except:
                        cph = CoxPHFitter(penalizer=0.1)
                        cph.fit(traindf, duration_col='BL2Target_yrs', event_col='target_y', formula=formula)
        y_pred = cph.predict_partial_hazard(testdf)
        y_pred_df = pd.DataFrame({'eid': testdf.eid, 'target_y': testdf.target_y, 'BL2Target_yrs': testdf.BL2Target_yrs, col_name + '_hazard': y_pred})
        y_out_df = pd.concat([y_out_df, y_pred_df], axis=0)
    y_out_df.sort_values(by = 'eid', ascending=True, inplace = True)
    return y_out_df[['eid', col_name + '_hazard']]

tgt_info_df = pd.read_csv(dpath + 'Data/TargetData/DiseaseTable.csv')
tgt_name_lst = tgt_info_df.loc[tgt_info_df[group + 'Analysis'] == 1].NAME.tolist()
tgt_name_lst.sort()

y_pred = pd.read_csv(dpath + 'Data/BloodData/BloodDataIndex.csv', usecols=['eid'])
cov_df = pd.read_csv(dpath + 'Data/Covariates/Covariates.csv')
cov_df['Race'].replace([1,2,3,4], [1, 0, 0, 0], inplace = True)

cov_fml = "Age + C(Sex) + C(Race) + TDI + BMI"
cov_fml_sex = "Age + C(Race) + TDI + BMI"


def process(dpath, cov_df, group, tgt_info_df, pair_omics_lst, cov_fml_sex, cov_fml, y_pred, tgt_name):
    try:
        sex_id = tgt_info_df.loc[tgt_info_df.NAME == tgt_name].SEX.iloc[0]
        for pair_omics in pair_omics_lst:
            omics0, omics1 = pair_omics[0], pair_omics[1]
            my_fml_omics = omics0 + ' + ' + omics1
            my_fml_omics_cov = \
            [my_fml_omics + ' + ' + cov_fml_sex if (sex_id == 1) | (sex_id == 2) else my_fml_omics + ' + ' + cov_fml][0]
            omics_traindf0, omics_testdf0 = get_omics_df(dpath, cov_df, group, omics0, tgt_name)
            omics_traindf1, omics_testdf1 = get_omics_df(dpath, cov_df, group, omics1, tgt_name)
            omics_traindf = pd.merge(omics_traindf0, omics_traindf1[['eid', omics1]], how='inner', on=['eid'])
            omics_testdf = pd.merge(omics_testdf0, omics_testdf1[['eid', omics1]], how='inner', on=['eid'])
            y_pred_omics = pred_hazard(omics_traindf, omics_testdf, my_fml_omics, omics0 + '_' + omics1)
            y_pred_omics_cov = pred_hazard(omics_traindf, omics_testdf, my_fml_omics_cov,
                                           omics0 + '_' + omics1 + '_Cov')
            y_pred = pd.merge(y_pred, y_pred_omics, how='inner', on=['eid'])
            y_pred = pd.merge(y_pred, y_pred_omics_cov, how='inner', on=['eid'])
        pair_omics = ['Metabolomic', 'Proteomic']
        omics0, omics1 = pair_omics[0], pair_omics[1]
        my_fml_omics = omics0 + ' + ' + omics1
        my_fml_omics_cov = \
            [my_fml_omics + ' + ' + cov_fml_sex if (sex_id == 1) | (sex_id == 2) else my_fml_omics + ' + ' + cov_fml][0]
        _, omics_df0 = get_omics_df(dpath, cov_df, group, omics0, tgt_name)
        _, omics_df1 = get_omics_df(dpath, cov_df, group, omics1, tgt_name)
        omics_df = pd.merge(omics_df0, omics_df1[['eid', omics1]], how='inner', on=['eid'])
        y_pred_omics = pred_hazard_cv(omics_df, fold_id_lst, my_fml_omics, omics0 + '_' + omics1)
        y_pred_omics_cov = pred_hazard_cv(omics_df, fold_id_lst, my_fml_omics_cov, omics0 + '_' + omics1 + '_Cov')
        y_pred = pd.merge(y_pred, y_pred_omics, how='inner', on=['eid'])
        y_pred = pd.merge(y_pred, y_pred_omics_cov, how='inner', on=['eid'])
        y_pred = pd.merge(omics_df[['eid', 'Split', 'target_y', 'BL2Target_yrs']], y_pred, how='inner', on=['eid'])
        y_pred.to_csv(dpath + 'Results/' + group + '/MultiOmics/RiskScoreRefit/Hazard/TwoOmics/' + tgt_name + '.csv', index=False)
    except:
        pass
    return None

out = Parallel(n_jobs=nb_cpus)(delayed(process)(dpath, cov_df, group, tgt_info_df, pair_omics_lst,
                                                cov_fml_sex, cov_fml, y_pred, tgt_name) for tgt_name in tgt_name_lst)




