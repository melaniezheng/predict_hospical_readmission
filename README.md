# predict_hospical_readmission

### Purpose
The purpose of this project is to develop a predictive model which will help hospitals
reduce their readmission rates among diabetic patients. <br>

### Introduction
A goal of the Affordable Health Care act is to increase the quality of hospital care in U.S hospitals. One specific issue area in which hospitals can improve quality is by improving hospital readmission rates. Under the Affordable Care Act, CMS created the Hospital Readmission Reduction Program in order to link payment data to the quality of hospital care in order to improve health quality for Americans. Essentially, payments to Inpatient Prospective Payment System (IPPS) hospitals depend on each hospital’s readmission rates. Hospitals with poor readmission performance are financially penalized through reduced payments.

#### Hospital Readmission Definitions according to CMS:
The 30-day risk standardized readmission measures include:
- All-cause unplanned readmissions that happen within 30 days of discharge from the index (i.e., initial) admission.
- Patients who are readmitted to the same hospital, or another applicable acute care hospital for any reason.

### Challenge
Develop a model which predicts whether a patient will be readmitted in under 30 days. A diabetic readmission reduction program intervention will use this model in order to target patients at high risk for readmission. Models will be evaluated on __AUC__, __TPR___ and  __FPR__.

### Deliverables
Model which predicts <30 readmission

### Data
The Clinical Database Patient Records includes 10 years (1999–2008) of clinical care data in 130 US hospitals pertaining to patients with diabetes. Research article: Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records also came contributed to digging deep into this dataset.

See data_prep.ipynb in current directory for data preperation efforts. The data cleaning steps has been organized into preprocess.py. 

### Why should we care?
The Hospital Readmissions Reduction Program (HRRP) is a Medicare value-based purchasing program that reduces payments to hospitals with excess readmissions. The program supports the national goal of improving healthcare for Americans by linking payment to the quality of hospital care.

From the hospital standpoint, it help both the hospital clinical care team and the finance team in different ways. For the clinical care team, they can target patients with higher readmission risk based on the model to provide extra care needed that could help mitigate readmission. The model also helps Finance team maximize the funding from the government as the result of the effort to reduce hospital readmission.
