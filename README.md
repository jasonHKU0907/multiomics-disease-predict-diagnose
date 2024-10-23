<div align="center">
  
![Logo](./src/CoverImage.png?raw=true "CoverImage")

</div>

## Description
A key challenge in precision medicine is optimizing risk stratification to enable timely diagnosis, prevention, and treatment, thereby improving patient outcomes. Blood tests stand out due to their advantages of invasiveness, accessibility, cost-effectiveness, and remarkable convenience. Multi-omics profiling offers a comprehensive panorama of human diseases, with the anticipation to inform the advancement of future clinical strategies for diagnosis, prediction, and intervention.

This repository contains python codes for model development and evaluations to the papaer "Integrative multi-omics blood profiling for human disease diagnosis and prediction". 
Detailed model performance and interative implementation can be access through website of https://multiomics-disease-predict-diagnose.com/

<div align="center">
  
![Workflow](./src/StudyOverview.png?raw=true "StudyOverview")

<div align="left">
  
## Study Overview
This webpage leveraged data from genomics, proteomics, metabolomics, and chemistry biomarkers to provide an implementation of multi-omics blood profiling in diagnosing 219 diseases and predicting 475 diseases. The risk models were established using a prospective cohort of UK Biobank (N=493,577) with long follow-up (median 14.6 years). This implementation was approved by UK Biobank under application number 19542 and 202239.

<div align="center">
  
![Architecture](./src/ModelingPipeline.png?raw=true "ModelingPipeline")

<div align="left">

## Modeling pipeline
The population was divided into omic-specified derivation cohorts for model development, employing a 10-fold cross-validation (CV) strategy. Specifically, nine folds of data served as the training set, while the remaining fold was used as the validation set. This process was iteratively repeated until each fold had been utilized for both training and validation. A hold-out test was conducted among 29,694 participants who had undergone comprehensive omics profiling. The machine learning pipeline encompassed the determination of biomarker panels and ensemble predictions, generating omic-specified risk scores—GenRS, MetRS, ProRS, and CheRS—for diagnostic and predictive purposes.
Downstream time-to-event and cross-sectional analyses were carried out using aggregated predictions from the validation sets within the omic-specified derivation cohorts. The 10-fold CV was executed using the same partitions as those used for risk score development. Evaluations were performed on the hold-out test. It is worth noting that, due to insufficient samples, 10-fold cross-validation was directly applied in the hold-out test for the analyses of ProRS+MetRS and GenRS+MetRS+ProRS+CheRS.
  
## License
This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].
[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]


## Citation   
```
You J., Liang Y., Chen Y.L., et al. Integrative multi-omics blood profiling for human disease diagnosis and prediction. under review
```
