

def add_variable(data):
    data['DAYS_EMPLOYED_PERC'] = data.loc[:,'DAYS_EMPLOYED'] / data.loc[:,'DAYS_BIRTH']
    data['INCOME_CREDIT_PERC'] = data.loc[:,'AMT_INCOME_TOTAL'] / data.loc[:,'AMT_CREDIT']
    data['INCOME_PER_PERSON'] = data['AMT_INCOME_TOTAL'] / data.loc[:,'CNT_FAM_MEMBERS']
    data['ANNUITY_INCOME_PERC'] = data.loc[:,'AMT_ANNUITY'] / data.loc[:,'AMT_INCOME_TOTAL']
    data['PAYMENT_RATE'] = data.loc[:,'AMT_ANNUITY'] / data.loc[:,'AMT_CREDIT']
    #return data


def get_client(id, data):
    col_index = len(data.columns)
    row_index =np.where(data['SK_ID_CURR'] == id)
    column_value_id = data.iloc[row_index[0][0], 1:col_index]
    df_column_value_id = column_value_id.to_frame().T
    return df_column_value_id