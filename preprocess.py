#### This is the data preparation effort to developing a Machine Learning model which predicts hospital readmission within 30 days (classification problem) among diabetic patients.
### Data Preprocess Steps:
    # 1. Drop encounter_id column as it doesn't provide any information for our objective.
    # 2. Re-map our target variable, Readmitted, into 2 labels. 1 for Readmitted within 30 days, 0 otherwise.
    # 3. Some missing values are encoded as ?, fill these in as NaN.
    # 4. weight, payer_code, and medical_specialty have over 1/3 of the values missing. Drop these columns.
    # 5. Recagtegorize three variables for diagnostic information, diag_1, diag_2, diag_3 into 9 categories as found in a research paper, https://www.hindawi.com/journals/bmri/2014/781670/tab2/ .
    # 6. Remove records where discharge_disposition_id = 11 which means 'Expired'.
    #    Remove records from the same patient based on rules defined by our understanding of data. See data_pre.ipynb for details.


import pandas as pd
import numpy as np

def clean_data():
    # load data
    seedn = 1234 
    def diag_group(x):
        # map diagnostic columns into 9 categories.
        try:
            x = float(x)
            if (x >= 390 and x <= 459) or x == 785:
                result = 'Circulatory'
            elif (x >= 460 and x <= 519) or x == 786:
                result = 'Respiratory'
            elif (x >= 520 and x <= 579) or x == 787:
                result = 'Digestive'
            elif x >= 250 and x < 251:
                result = 'Diabetes'
            elif x >= 800 and x <= 999:
                result = 'Injury'
            elif x >=710 and x <= 739:
                result = 'Musculoskeletal'
            elif (x >= 580 and x <=629) or x == 788:
                result = 'Genitourinary'
            elif (x >= 140 and x <= 239):
                result = 'Neoplasms'
            else:
                result = 'Other'
        except:
            result = 'Other'
        return result

    data = pd.read_csv('./data/diabetic_data.csv')
    df = data.drop(columns=['encounter_id']) # drop id column
    df['readmitted'] = df.readmitted.apply(lambda x: 1 if x == '<30' else 0) # binarize readmitted (target variable)

    df = df.replace(r'\?', np.nan, regex=True) # missing values
    df.drop(columns = ['weight','payer_code','medical_specialty'], inplace = True) 

    df['diag_1'] = df['diag_1'].apply(lambda x: np.nan if x is np.nan else diag_group(x))
    df['diag_2'] = df['diag_2'].apply(lambda x: np.nan if x is np.nan else diag_group(x))
    df['diag_3'] = df['diag_3'].apply(lambda x: np.nan if x is np.nan else diag_group(x))

    df = df[df['discharge_disposition_id']!=11] # drop expired discharge disposition ids.


    ################################ Drop duplicated patients ################################

    # 3 types of duplicated patients:
    # 1) Same patient has multiple entries (>=2 entries) and 1 of the record shows readmitted.
    # i.e. patient_nbr = 31320.
    # Keep the 1 record where readmitted is 1.

    # 2) Same patient has multiple entries (>=2 entries) and readmitted more than 1 time.
    # i.e. patient_nbr = 88785891.
    # Keep the record where readmitted is 1 and having highest time_in_hospital. Assuming the time_in_hospital is a good indicator of readmission.
    # If multiple entries has the highest time_in_hospital, keep one record at random.

    # 3) Same patient having more than one entries but have no readmitted record.
    # i.e. patient_nbr = 189257846.
    # Drop records at random, leaving only 1 record for each patient.
    dupe_patients=df[df.duplicated('patient_nbr', False)].sort_values(by='patient_nbr')
    a = dupe_patients.groupby('patient_nbr').size().reset_index().rename(columns={0:'entry_count'})
    b = dupe_patients.groupby('patient_nbr').sum()[['readmitted']].reset_index()
    dupes = pd.merge(a,b,on = 'patient_nbr')

    patients_manyEntries = dupes.patient_nbr.tolist() # all unique patient_numbers
    patients_readmittedOnce = dupes[dupes['readmitted']==1]['patient_nbr'].tolist() # scenario 1
    patients_readmittedMany = dupes[dupes['readmitted']>1]['patient_nbr'].tolist() # scenario 2
    patients_notreadmitted = dupes[dupes['readmitted']==0]['patient_nbr'].tolist() # scenario 3
    
    def drop_patients_readmittedOnce(patient_num): 
        tmp = df[np.logical_and(df['patient_nbr']==patient_num,df['readmitted']==1)]
        return tmp.values.tolist()[0]

    def drop_patients_readmittedMany(patient_num):
        tmp = df[np.logical_and(df['patient_nbr']==patient_num,df['readmitted']==1)]
        tmp = tmp[tmp['time_in_hospital']==max(tmp['time_in_hospital'])]
        np.random.seed(seedn)
        idx = np.random.permutation(np.arange(len(tmp)))
        tmp = tmp.iloc[idx].drop_duplicates('patient_nbr')
        return tmp.values.tolist()[0]

    def drop_patients_notreadmitted(patient_num):
        tmp = df[df['patient_nbr']==patient_num]
        np.random.seed(seedn)
        idx = np.random.permutation(np.arange(len(tmp)))
        tmp = tmp.iloc[idx].drop_duplicates('patient_nbr')
        return tmp.values.tolist()[0]

    list_once = []
    for n in patients_readmittedOnce:
        list_once.append(drop_patients_readmittedOnce(n))
    list_many = []
    for n in patients_readmittedMany:
        list_many.append(drop_patients_readmittedMany(n))
    list_none = []
    for n in patients_notreadmitted:
        list_none.append(drop_patients_notreadmitted(n))

    add = pd.concat([pd.DataFrame(list_once), pd.DataFrame(list_many), pd.DataFrame(list_none)], axis=0)
    add.columns = df.columns
    df = df[~df['patient_nbr'].isin(patients_manyEntries)] # drop all duplicated patients from dataset.
    df = pd.concat([df, add], axis=0) # add back unique patient records

    for col in ['time_in_hospital','number_inpatient','number_diagnoses']:
        df[col] = df[col].apply(lambda x: 10 if x > 10 else x)

    df.reset_index(drop=True, inplace = True)
    
    # drop_categorical_columns = ['acetohexamide', 'troglitazone', 'examide', 'citoglipton', 'glipizide-metformin', 'glimepiride-pioglitazone', 'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone']
    # df.drop(columns=drop_categorical_columns, inplace = True)

    df = df[df.gender.isin(['Female','Male'])] # drop 3 obs where gender is Unknown/Invalid

    df.to_csv('df_preprocessed.csv', index=False)
    return df