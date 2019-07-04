import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import math

columns_list = ['loan_amnt', 'funded_amnt','total_pymnt',
                'term', 'int_rate', #'installment',
                'emp_length', 'home_ownership', 'annual_inc',
                'verification_status', 'loan_status',
                'purpose', 'zip_code', 'addr_state', 'dti',
                'delinq_2yrs', 'earliest_cr_line', 'fico_range_low',
                'fico_range_high', 'inq_last_6mths', 'mths_since_last_delinq',
                'mths_since_last_record', 'open_acc', 'pub_rec', 'revol_bal',
                'revol_util', 'total_acc', #'out_prncp',
                #'out_prncp_inv', 
                'last_fico_range_high', 'last_fico_range_low',
                'collections_12_mths_ex_med', 'mths_since_last_major_derog',
                'policy_code', 'application_type', 'annual_inc_joint', 'dti_joint',
                'acc_now_delinq', 'tot_coll_amt',
                #'tot_cur_bal', 
                'open_acc_6m', #'open_act_il', 
                #'open_il_12m',
                #'open_il_24m', #'mths_since_rcnt_il', 
                #'total_bal_il', 'il_util',
                'open_rv_12m', 'open_rv_24m', 'all_util',
                'total_rev_hi_lim', 'inq_fi', 'total_cu_tl', 'inq_last_12m',
                'acc_open_past_24mths', #'avg_cur_bal', 
                'bc_open_to_buy', 'bc_util',
                'chargeoff_within_12_mths', 'delinq_amnt', 'mo_sin_old_il_acct',
                'mo_sin_old_rev_tl_op', 'mo_sin_rcnt_rev_tl_op', #'mo_sin_rcnt_tl',
                'mort_acc', 'mths_since_recent_bc', 'mths_since_recent_bc_dlq',
                #'mths_since_recent_inq', 
                'mths_since_recent_revol_delinq',
                'num_accts_ever_120_pd', 'num_actv_bc_tl', 'num_actv_rev_tl',
                'num_bc_sats', 'num_bc_tl', 'num_il_tl', 'num_op_rev_tl',
                'num_rev_accts', 'num_rev_tl_bal_gt_0', 'num_sats',
                'num_tl_120dpd_2m', 'num_tl_30dpd', 'num_tl_90g_dpd_24m',
                'num_tl_op_past_12m', 'pct_tl_nvr_dlq', 'percent_bc_gt_75',
                'pub_rec_bankruptcies', 'tax_liens', 'tot_hi_cred_lim',
                'total_bal_ex_mort', 'total_bc_limit',
                'total_il_high_credit_limit','grade', 'collection_recovery_fee', 
                #'total_rec_prncp', 
                'title', #'total_rec_int', 'total_rec_late_fee', 
                'sub_grade', 'debt_settlement_flag', 'emp_title','issue_d','last_pymnt_d']

dtype = {
      'loan_amnt': 'int64',
      'term': 'object',
      'int_rate': 'float64',
      'emp_length': 'object',
      'home_ownership': 'object',
      'annual_inc': 'float64',
      'verification_status': 'object',
      'loan_status': 'object',
      'purpose': 'object',
      'zip_code': 'object',
      'addr_state': 'object',
      'dti': 'float64',
      'delinq_2yrs': 'int64',
      'fico_range_low': 'int64',
      'fico_range_high': 'int64',
      'inq_last_6mths': 'int64',
      'open_acc': 'int64',
      'pub_rec': 'int64',
      'revol_bal': 'int64',
      'total_acc': 'int64',
      'application_type': 'object'
}

nan_max_cols = [
    'mths_since_last_delinq',
    'mths_since_last_record',
    'mths_since_last_major_derog',
    'mths_since_rcnt_il',
    'mths_since_recent_bc',
    'mths_since_recent_bc_dlq',
    'mths_since_recent_inq',
    'mths_since_recent_revol_delinq',
    'mo_sin_old_il_acct',
    'mo_sin_old_rev_tl_op',
    'mo_sin_rcnt_rev_tl_op',
    'mo_sin_rcnt_tl',
]

nan_zero_cols = [
    'tot_coll_amt',
    'tot_cur_bal',
    'open_acc_6m',
    'open_act_il',
    'open_il_12m',
    'open_il_24m',
    'total_bal_il',
    'open_rv_12m',
    'open_rv_24m',
    'inq_fi',
    'inq_last_12m',
    'num_accts_ever_120_pd',
    'num_actv_bc_tl',
    'num_actv_rev_tl',
    'num_bc_sats',
    'num_bc_tl',
    'num_il_tl',
    'num_op_rev_tl',
    'num_rev_accts',
    'num_rev_tl_bal_gt_0',
    'num_sats',
    'num_tl_120dpd_2m',
    'num_tl_30dpd',
    'num_tl_90g_dpd_24m',
    'num_tl_op_past_12m',
    'mort_acc',
    'percent_bc_gt_75',
    'tot_hi_cred_lim',
    'total_bal_ex_mort',
    'total_bc_limit',
    'total_il_high_credit_limit',
    'bc_open_to_buy',
    'acc_open_past_24mths',
    'total_cu_tl',
    'total_rev_hi_lim',
    'emp_length',
    'pub_rec_bankruptcies',
    'tax_liens'
]

nan_mean_cols = [
    'dti',
    'annual_inc_joint',
    'dti_joint',
    'il_util',
    'all_util',
    'bc_util',
    'pct_tl_nvr_dlq',
    'avg_cur_bal',
    'revol_util',
]


############ DATA CLEANING

###IMPUTE FUNCTIONS

def impute_means_zeros_maxs_X_train_X_test(X_train, X_test, nan_max_cols, nan_zero_cols, nan_mean_cols):
    '''Impute means, zeros, and maxs in X_train & X_test based on X_train values, 
    given column list for each impute type'''
    for col in X_train:
        dt = X_train[col].dtype
        #FILL MISSING WITH MEAN from X_train, in both X_train & X_test
        if col in nan_mean_cols:
            X_train[col] = X_train[col].fillna(np.nanmean(X_train[col].values))
            X_test[col] = X_test[col].fillna(np.nanmean(X_train[col].values))
        #FILL MISSING WITH ZEROS from X_train, in both X_train & X_test
        if col in nan_zero_cols:
            X_train[col] = X_train[col].fillna(0.0)
            X_test[col] = X_test[col].fillna(0.0)
        #FILL MISSING WITH MAX from X_train, in both X_train & X_test
        if col in nan_max_cols:
            X_train[col] = X_train[col].fillna(np.nanmax(X_train[col].values) * 5)
            X_test[col] = X_test[col].fillna(np.nanmax(X_train[col].values) * 5)
    return X_train, X_test

def impute_means_zeros_maxs_X(X, nan_max_cols, nan_zero_cols, nan_mean_cols):
    '''Impute means, zeros, and maxs in X given column list for each impute type'''
    for col in X:
        dt = X[col].dtype
        #FILL MISSING WITH MEAN from X
        if col in nan_mean_cols:
            X[col] = X[col].fillna(np.nanmean(X[col].values))
        #FILL MISSING WITH ZEROS from X
        if col in nan_zero_cols:
            X[col] = X[col].fillna(0.0)
        #FILL MISSING WITH MAX from X
        if col in nan_max_cols:
            X[col] = X[col].fillna(np.nanmax(X[col].values) * 5)
    return X

### PERCENTAGE PARSING

def parse_percentage(percentage):
    '''Change percentage features into floats'''
    if str(percentage) == 'nan': return math.nan
    new = percentage.replace('%', '')
    return float(new) / 100.0

### CLEANING FUNCTIONS
def view_columns_by_number_of_rows_that_have_nan(X):
    null_columns=X.columns[X.isnull().any()]
    output = X[null_columns].isnull().sum()
    return output

def clean_LC_data_classification_eval(dfs_list):
    '''Prepare completed loan term LendingClub data for classification to compare to labels.
    Returns clean DataFrame ready for model-based EVALUATION'''
    raw_lc_df = pd.concat(dfs_list, ignore_index=True).set_index('id')
    # Uses completed loans
    raw_lc_df = raw_lc_df.loc[(raw_lc_df['loan_status'] == 'Charged Off') |
                              (raw_lc_df['loan_status'] == 'Fully Paid') |
                              (raw_lc_df['loan_status'] == 'Default'),:]
    raw_lc_df['loan_status'] = raw_lc_df['loan_status'].map({'Charged Off': 0, 'Default': 0, 'Fully Paid': 1})
    ###
    raw_lc_df['earliest_cr_line'] = pd.to_timedelta(pd.to_datetime(raw_lc_df['earliest_cr_line'])).dt.days
    raw_lc_df['revol_util'] = raw_lc_df['revol_util'].apply(parse_percentage)
    raw_lc_df['int_rate'] = raw_lc_df['int_rate'].apply(parse_percentage)
    lc_df = raw_lc_df[columns_list]
    lc_df = lc_df.dropna(axis=0, subset=['loan_amnt','inq_last_6mths'])
    #lc_df.set_index('id',inplace=True)
    lc_df = lc_df.astype(dtype=dtype)
    lc_df.loc[lc_df['emp_length'] == '< 1 year','emp_length'] = '0'
    lc_df.loc[lc_df['emp_length'] == '10+ years', 'emp_length'] = '10'
    lc_df['emp_length'] = lc_df['emp_length'].str[:1]
    lc_df['emp_length'] = lc_df['emp_length'].astype(float)
    lc_df = lc_df[lc_df['zip_code'].notnull()]
    lc_df = lc_df[lc_df['emp_title'].notnull()]
    #lc_df.set_index('id',inplace=True)
    counts = lc_df['emp_title'].value_counts()
    idx = counts[counts.lt(1600)].index
    lc_df.loc[lc_df['emp_title'].isin(idx) == False, 'emp_title_2'] = lc_df['emp_title']
    lc_df.loc[lc_df['emp_title'].isin(idx), 'emp_title_2'] = 'Other'
    clean_lc_df = lc_df.dropna(subset=
                               ['collections_12_mths_ex_med',
                                'chargeoff_within_12_mths','last_pymnt_d'],axis=0)
    return clean_lc_df


def clean_new_LC_data_classification_current(dfs_list):
    '''Prepare new, current, investable LendingClub data for classification to make recommendations. 
    Returns clean DataFrame ready for model-based RECOMMENDATION'''
    raw_lc_df = pd.concat(dfs_list, ignore_index=True).set_index('id')
    # Uses current loans
    raw_lc_df = raw_lc_df.loc[raw_lc_df['loan_status'] == 'Current',:]
    #raw_lc_df.drop(columns=['loan_status'], inplace=True)
    ###
    raw_lc_df['earliest_cr_line'] = pd.to_timedelta(pd.to_datetime(raw_lc_df['earliest_cr_line'])).dt.days
    raw_lc_df['revol_util'] = raw_lc_df['revol_util'].apply(parse_percentage)
    raw_lc_df['int_rate'] = raw_lc_df['int_rate'].apply(parse_percentage)
    lc_df = raw_lc_df[columns_list]
    lc_df = lc_df.dropna(axis=0, subset=['loan_amnt','inq_last_6mths'])
    lc_df = lc_df.astype(dtype=dtype)
    lc_df.loc[lc_df['emp_length'] == '< 1 year','emp_length'] = '0'
    lc_df.loc[lc_df['emp_length'] == '10+ years', 'emp_length'] = '10'
    lc_df['emp_length'] = lc_df['emp_length'].str[:1]
    lc_df['emp_length'] = lc_df['emp_length'].astype(float)
    lc_df = lc_df[lc_df['zip_code'].notnull()]
    lc_df = lc_df[lc_df['emp_title'].notnull()]
    #lc_df.set_index('id',inplace=True)
    counts = lc_df['emp_title'].value_counts()
    idx = counts[counts.lt(1600)].index
    lc_df.loc[lc_df['emp_title'].isin(idx) == False, 'emp_title_2'] = lc_df['emp_title']
    lc_df.loc[lc_df['emp_title'].isin(idx), 'emp_title_2'] = 'Other'
    clean_lc_df_current = lc_df.dropna(subset=
                                       ['collections_12_mths_ex_med',
                                        'chargeoff_within_12_mths','last_pymnt_d'],axis=0)
    return clean_lc_df_current

######### PREPROCESSING

### State predictors and targets

def preprocessing_eval(clean_lc_df):
    '''Initiate X_train, X_test, y_train, y_test and impute missing values'''
    X = clean_lc_df.drop(columns=['loan_status'])
    y = clean_lc_df[['loan_status']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0,test_size=0.3)
    X_train.drop(columns=['title'],inplace=True)
    X_test.drop(columns=['title'],inplace=True)
    # CALL IMPUTE FUNCTION on X_train, X_test
    X_train, X_test = impute_means_zeros_maxs_X_train_X_test(X_train, X_test, nan_max_cols, nan_zero_cols, nan_mean_cols)
    return X_train, X_test, y_train, y_test

def preprocessing_future_test(clean_lc_df_future):
    '''Initiate X & Y and impute missing values for Completed loans'''
    X_future = clean_lc_df_future.drop(columns=['loan_status'])
    y_future = clean_lc_df_future[['loan_status']]
    y_future['loan_status'] = y_future['loan_status'].astype(int)
    #pd.DataFrame(clean_lc_df_future['loan_status'], index=clean_lc_df_future.index, columns=['class_pred'])
    X_future.drop(columns=['title'],inplace=True) 
    # CALL IMPUTE FUNCTION on X_current
    X_future = impute_means_zeros_maxs_X(X_future, nan_max_cols, nan_zero_cols, nan_mean_cols)
    return X_future, y_future

def preprocessing_current(clean_lc_df_current):
    '''Initiate X & Y and impute missing values for Current loans'''
    X_current = clean_lc_df_current.drop(columns=['loan_status'])
    y_current = pd.DataFrame(np.nan, index=clean_lc_df_current.index, columns=['prob_default'])
    X_current.drop(columns=['title'],inplace=True) 
    # CALL IMPUTE FUNCTION on X_current
    X_current = impute_means_zeros_maxs_X(X_current, nan_max_cols, nan_zero_cols, nan_mean_cols)
    return X_current, y_current

###### One Hot Encoding

#Load X_train for use in One Hot Encoding
X_train_pre_ohe_for_future_encoder = pd.read_pickle('X_train_pre_ohe_for_future_encoder.pkl')

def one_hot_encode_eval(X_train, X_test):
    '''One Hot Encoder for 6x categorical vars on X_train, transforming X_train & X_test'''
    encoder = OneHotEncoder(categories='auto',handle_unknown='ignore').fit(X_train[['home_ownership']])
    ohe_home_ownership = pd.DataFrame(encoder.transform(X_train[['home_ownership']]).toarray(),
                                      columns=encoder.get_feature_names(["home_ownership"]),index=X_train.index)
    ohe_home_ownership_test = pd.DataFrame(encoder.transform(X_test[['home_ownership']]).toarray(),
                                           columns=encoder.get_feature_names(["home_ownership"]),index=X_test.index)
    encoder = OneHotEncoder(categories='auto',handle_unknown='ignore').fit(X_train[['purpose']])
    ohe_purpose = pd.DataFrame(encoder.transform(X_train[['purpose']]).toarray(),
                               columns=encoder.get_feature_names(["purpose"]),index=X_train.index)
    ohe_purpose_test = pd.DataFrame(encoder.transform(X_test[['purpose']]).toarray(),
                                    columns=encoder.get_feature_names(["purpose"]),index=X_test.index)
    encoder = OneHotEncoder(categories='auto',handle_unknown='ignore').fit(X_train[['zip_code']])
    ohe_zip_code = pd.DataFrame(encoder.transform(X_train[['zip_code']]).toarray(),
                                columns=encoder.get_feature_names(["zip_code"]),index=X_train.index)
    ohe_zip_code_test = pd.DataFrame(encoder.transform(X_test[['zip_code']]).toarray(),
                                     columns=encoder.get_feature_names(["zip_code"]),index=X_test.index)
    encoder = OneHotEncoder(categories='auto',handle_unknown='ignore').fit(X_train[['application_type']])
    ohe_application_type = pd.DataFrame(encoder.transform(X_train[['application_type']]).toarray(),
                                        columns=encoder.get_feature_names(["application_type"]),index=X_train.index)
    ohe_application_type_test = pd.DataFrame(encoder.transform(X_test[['application_type']]).toarray(),
                                             columns=encoder.get_feature_names(["application_type"]),index=X_test.index)
    encoder = OneHotEncoder(categories='auto',handle_unknown='ignore').fit(X_train[['sub_grade']])
    ohe_sub_grade = pd.DataFrame(encoder.transform(X_train[['sub_grade']]).toarray(),
                                 columns=encoder.get_feature_names(["sub_grade"]),index=X_train.index)
    ohe_sub_grade_test = pd.DataFrame(encoder.transform(X_test[['sub_grade']]).toarray(),
                                      columns=encoder.get_feature_names(["sub_grade"]),index=X_test.index)
    encoder = OneHotEncoder(categories='auto',handle_unknown='ignore').fit(X_train[['emp_title_2']])
    ohe_emp_title_2 = pd.DataFrame(encoder.transform(X_train[['emp_title_2']]).toarray(),
                                   columns=encoder.get_feature_names(["emp_title_2"]),index=X_train.index)
    ohe_emp_title_2_test = pd.DataFrame(encoder.transform(X_test[['emp_title_2']]).toarray(),
                                        columns=encoder.get_feature_names(["emp_title_2"]),index=X_test.index)
    return (ohe_home_ownership, ohe_purpose, ohe_zip_code, ohe_application_type, ohe_sub_grade, ohe_emp_title_2, 
            ohe_home_ownership_test, ohe_purpose_test, ohe_zip_code_test, ohe_application_type_test,
            ohe_sub_grade_test, ohe_emp_title_2_test)
    
def one_hot_encode_current(X_current):
    '''One Hot Encoder for 6x categorical vars on X_train, transforming X_current'''
    encoder = OneHotEncoder(categories='auto',handle_unknown='ignore').fit(
        X_train_pre_ohe_for_future_encoder[['home_ownership']])
    ohe_home_ownership = pd.DataFrame(encoder.transform(X_current[['home_ownership']]).toarray(),
                                      columns=encoder.get_feature_names(["home_ownership"]),index=X_current.index)
    encoder = OneHotEncoder(categories='auto',handle_unknown='ignore').fit(
        X_train_pre_ohe_for_future_encoder[['purpose']])
    ohe_purpose = pd.DataFrame(encoder.transform(X_current[['purpose']]).toarray(),
                               columns=encoder.get_feature_names(["purpose"]),index=X_current.index)
    encoder = OneHotEncoder(categories='auto',handle_unknown='ignore').fit(
        X_train_pre_ohe_for_future_encoder[['zip_code']])
    ohe_zip_code = pd.DataFrame(encoder.transform(X_current[['zip_code']]).toarray(),
                                columns=encoder.get_feature_names(["zip_code"]),index=X_current.index)
    encoder = OneHotEncoder(categories='auto',handle_unknown='ignore').fit(
        X_train_pre_ohe_for_future_encoder[['application_type']])
    ohe_application_type = pd.DataFrame(encoder.transform(X_current[['application_type']]).toarray(),
                                        columns=encoder.get_feature_names(["application_type"]),index=X_current.index)
    encoder = OneHotEncoder(categories='auto',handle_unknown='ignore').fit(
        X_train_pre_ohe_for_future_encoder[['sub_grade']])
    ohe_sub_grade = pd.DataFrame(encoder.transform(X_current[['sub_grade']]).toarray(),
                                 columns=encoder.get_feature_names(["sub_grade"]),index=X_current.index)
    encoder = OneHotEncoder(categories='auto',handle_unknown='ignore').fit(
        X_train_pre_ohe_for_future_encoder[['emp_title_2']])
    ohe_emp_title_2 = pd.DataFrame(encoder.transform(X_current[['emp_title_2']]).toarray(),
                                   columns=encoder.get_feature_names(["emp_title_2"]),index=X_current.index)
    return (ohe_home_ownership, ohe_purpose, ohe_zip_code, ohe_application_type, ohe_sub_grade, ohe_emp_title_2)

def concat_X_and_6ohe_dfs(X, ohe_home_ownership, ohe_purpose, ohe_zip_code, 
                          ohe_application_type, ohe_sub_grade, ohe_emp_title_2):
    '''Concatenate one-hot encoded dataframes into a single dataframe'''
    X_comb = X
    X_comb = pd.concat([X_comb.drop("home_ownership", axis=1), ohe_home_ownership], axis=1)

    X_comb = pd.concat([X_comb.drop("purpose", axis=1), ohe_purpose], axis=1)

    X_comb = pd.concat([X_comb.drop("zip_code", axis=1), ohe_zip_code], axis=1)

    X_comb = pd.concat([X_comb.drop("application_type", axis=1), ohe_application_type], axis=1)

    X_comb = pd.concat([X_comb.drop("sub_grade", axis=1), ohe_sub_grade], axis=1)

    X_comb = pd.concat([X_comb.drop("emp_title_2", axis=1), ohe_emp_title_2], axis=1)
    return X_comb

##### Classification Prep
def prep_all_df_for_classification(X_all_df):
    '''drop OHE source columns & unuseful categorical variables'''
    X_all_df.drop(columns=['term','verification_status',
                           'grade','emp_title', 'addr_state',
                           #ALSO, drop redundant columns that new OHE columns provide the info for
                           'debt_settlement_flag',#ALSO, drop columns clearly not predictive of class
                           'issue_d','last_pymnt_d'],inplace=True) #ALSO, drop date columns
#call function pre-OHE
prep_all_df_for_classification(X_train_pre_ohe_for_future_encoder)


##### Regression Prep
def calc_annu_return(input_df):
    '''Calculates annual return given cleaned LendingClub dataframe'''
    annu_return = (input_df['total_pymnt'] /
                   input_df['funded_amnt']) ** (365.0 /
                                                ((pd.to_datetime(input_df['last_pymnt_d']) - 
                                                  pd.to_datetime(input_df['issue_d'])).dt.days + 30)) - 1
    return annu_return

def impute_annu_return_to_y(input_df, target_df):
    '''Use input dataframe to compute annualized return and add to target dataframe'''
    target_df['annu_return'] = calc_annu_return(input_df)
    target_df_return = target_df[['annu_return']]
    return target_df_return, target_df

def prep_df_for_regression_current(X_all_df):
    '''drop OHE source columns & unuseful categorical variables'''
    X_all_df.drop(columns=['term','verification_status',
                           'grade','emp_title', 'addr_state',
                           #ALSO, drop redundant columns that new OHE columns provide the info for
                           'debt_settlement_flag',#ALSO, drop columns clearly not predictive of class
                           'issue_d','last_pymnt_d'],inplace=True) #ALSO, drop date columns

def prep_df_for_regression_eval(X_train_all,X_test_all):
    '''drop OHE source columns & unuseful categorical variables'''
    X_train_all.drop(columns=['term','verification_status',
                          'grade','emp_title', 'addr_state',
                          #ALSO, drop redundant columns that new OHE columns provide the info for
                          'debt_settlement_flag',#ALSO, drop columns clearly not predictive of class
                          'issue_d','last_pymnt_d'],inplace=True) #ALSO, drop date columns
    X_test_all.drop(columns=['term','verification_status',
                          'grade','emp_title', 'addr_state',
                          #ALSO, drop redundant columns that new OHE columns provide the info for
                          'debt_settlement_flag',#ALSO, drop columns clearly not predictive of class
                          'issue_d','last_pymnt_d'],inplace=True) #ALSO, drop date columns
    
def scale_current(X_all_df):
    '''Scale current data to prep for regression prediction'''
    scaler = StandardScaler()
    scaler.fit(X_all_df.values)
    X_all_df_scaled = scaler.transform(X_all_df.values)
    X_all_df_scaled = pd.DataFrame(X_all_df_scaled, index=X_all_df.index, columns=X_all_df.columns)
    return X_all_df_scaled

def scale_eval(X_train_all,X_test_all):
    '''Scale train & test data to prep for regression prediction/evaluation'''
    scaler = StandardScaler()
    scaler.fit(X_train_all)
    X_train_all_scaled = scaler.transform(X_train_all)
    X_test_all_scaled = scaler.transform(X_test_all)
    return X_train_all_scaled, X_test_all_scaled

##### RUNNING PREDICTIONS ON CURRENT DATA

def classification_model_eval_prep_pipeline(dfs_list):
    ### CLASSIFICATION PREP
    clean_lc_df_eval = clean_LC_data_classification_eval(dfs_list)
    X_train, X_test, y_train_classif, y_test_classif = preprocessing_eval(clean_lc_df_eval)
    (ohe_home_ownership, ohe_purpose, ohe_zip_code, ohe_application_type, ohe_sub_grade, ohe_emp_title_2,
    ohe_home_ownership_test, ohe_purpose_test, ohe_zip_code_test, ohe_application_type_test, 
    ohe_sub_grade_test, ohe_emp_title_2_test) = one_hot_encode_eval(X_train, X_test)
    X_train_classif = concat_X_and_6ohe_dfs(X_train, ohe_home_ownership, ohe_purpose, ohe_zip_code, 
                                            ohe_application_type, ohe_sub_grade, ohe_emp_title_2)
    X_test_classif = concat_X_and_6ohe_dfs(X_test, ohe_home_ownership_test, ohe_purpose_test, ohe_zip_code_test, 
                                           ohe_application_type_test, ohe_sub_grade_test, ohe_emp_title_2_test)
    prep_all_df_for_classification(X_train_classif)#drops columns in place
    prep_all_df_for_classification(X_test_classif)#drops columns in place
    return (X_train_classif, X_test_classif, y_train_classif, y_test_classif) 
            

def regression_model_eval_prep_pipeline(dfs_list):
    clean_lc_df_eval = clean_LC_data_classification_eval(dfs_list)
    X_train, X_test, y_train_classif, y_test_classif = preprocessing_eval(clean_lc_df_eval)
    (ohe_home_ownership, ohe_purpose, ohe_zip_code, ohe_application_type, ohe_sub_grade, ohe_emp_title_2,
    ohe_home_ownership_test, ohe_purpose_test, ohe_zip_code_test, ohe_application_type_test, 
    ohe_sub_grade_test, ohe_emp_title_2_test) = one_hot_encode_eval(X_train, X_test)
    X_train_regr = concat_X_and_6ohe_dfs(X_train, ohe_home_ownership, ohe_purpose, ohe_zip_code, 
                                         ohe_application_type, ohe_sub_grade, ohe_emp_title_2)
    X_test_regr = concat_X_and_6ohe_dfs(X_test, ohe_home_ownership_test, ohe_purpose_test, ohe_zip_code_test, 
                                        ohe_application_type_test, ohe_sub_grade_test, ohe_emp_title_2_test)
    ### REGRESSION PREP
    y_train_regr, y_train = impute_annu_return_to_y(X_train_regr,y_train_classif)
    y_test_regr, y_test = impute_annu_return_to_y(X_test_regr,y_train_classif)
    prep_df_for_regression_eval(X_train_regr,X_test_regr) #drops columns in place
    X_train_regr_scaled, X_test_regr_scaled = scale_eval(X_train_regr,X_test_regr)
    return (X_train_regr, X_test_regr, y_train_regr, y_test_regr)


def current_pipeline(dfs_list):
    #CLASSIFICATION PIPELINE
    clean_lc_df_current = clean_new_LC_data_classification_current(dfs_list)
    X_current, y_current = preprocessing_current(clean_lc_df_current)
    (ohe_home_ownership, ohe_purpose, ohe_zip_code, 
     ohe_application_type, ohe_sub_grade, ohe_emp_title_2) = one_hot_encode_current(X_current)
    X_current_classif = concat_X_and_6ohe_dfs(X_current, ohe_home_ownership, ohe_purpose, ohe_zip_code,
                                          ohe_application_type, ohe_sub_grade, ohe_emp_title_2)
    X_current_regr = concat_X_and_6ohe_dfs(X_current, ohe_home_ownership, ohe_purpose, ohe_zip_code, 
                                       ohe_application_type, ohe_sub_grade, ohe_emp_title_2)
    prep_all_df_for_classification(X_current_classif) #drops columns in place
    loaded_log_reg_v1 = joblib.load('log_reg_v1.joblib')
    current_class_preds_proba = loaded_log_reg_v1.predict_proba(X_current_classif)
    y_current['prob_default'] = current_class_preds_proba[:,0]
    #REGRESSION PIPELINE
    y_current_regr, y_current = impute_annu_return_to_y(X_current_regr,y_current)
    prep_df_for_regression_current(X_current_regr)
    X_current_regr_scaled = scale_current(X_current_regr)
    loaded_ridge_reg_v1 = joblib.load('ridge_lin_reg_v1.joblib')
    current_return_preds = loaded_ridge_reg_v1.predict(X_current_regr_scaled)
    # Connecting
    y_predictions = y_current
    y_predictions['return_preds'] = current_return_preds
    y_predictions.drop(columns=['annu_return'],inplace=True)
    final_df = y_predictions.join(X_current_regr)
    return final_df

def show_recommendation_table(final_df):
    return rec_table

def show_recommendation_summary(rec_table):
    return

def plot_recommendation_visuals(rec_table):
    return

    
    
    