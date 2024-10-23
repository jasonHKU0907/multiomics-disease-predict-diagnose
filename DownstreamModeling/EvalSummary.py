
import glob
import re
import pandas as pd
from tqdm import tqdm

group = 'Incident'
#dpath = '/home1/jiayou/Documents/Projects/BloodOmicsPred/'
dpath = '/Volumes/JasonWork/Projects/BloodOmicsPred/'

def sort_nicely(l):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c.replace("_","")) for c in re.split('([0-9]+)', key) ]
    l.sort( key=alphanum_key )
    return l

tgt_name_df = pd.read_csv(dpath + 'Data/TargetData/DiseaseTable.csv')
tgt_name_lst = tgt_name_df.loc[tgt_name_df[group + 'Analysis'] == 1].NAME.tolist()
tgt_name_lst.sort()
tgt_dict_df = pd.read_csv(dpath + 'Data/TargetData/DiseaseTable.csv', usecols = ['NAME', 'LONGNAME', 'Root', 'ICD_10'])

tmp_df = pd.read_csv(dpath + 'Results/' + group + '/MultiOmics/RiskScoreRefit/Evaluation/E4_DM2.csv')
mod_lst = tmp_df.iloc[:,0].tolist()
mod_lst = [ele.split('_hazard')[0] for ele in mod_lst]

eval_df = pd.DataFrame()

for tgt_name in tqdm(tgt_name_lst):
    tgt_eval_df = pd.read_csv(dpath + 'Results/' + group + '/MultiOmics/RiskScoreRefit/Evaluation/'+tgt_name+'.csv')
    tgt_eval_df[tgt_name] = tgt_eval_df.C_index
    eval_df = pd.concat([eval_df, tgt_eval_df[[tgt_name]]], axis = 1)

eval_df.index = mod_lst
eval_df = eval_df.T
eval_df['NAME'] = eval_df.index
eval_df = pd.merge(eval_df, tgt_dict_df, how = 'left', on = ['NAME'])

eval_df.to_csv(dpath + 'Results/' + group + '/MultiOmics/RiskScoreRefit/Hazard_AUC_Summary.csv', index = True)


