
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 16:39:04 2020
@author:
"""
# 1. Library imports
import uvicorn
from fastapi import FastAPI
import numpy as np
import pickle
import pandas as pd
#from processing_functions import add_variable

# 2. Create the app object
app = FastAPI()

#data_train = pd.read_csv('application_train.csv')

#import model and data treatment process
#pickle_in = open("pipeline_bank.pkl","rb")
#pipeline_process=pickle.load(pickle_in)

#add_variable(data_train)

@app.post('/')
def get_name(name: str):
    id = request.args.get("id")
    return {'Welcome To Krish Youtube Channel': f'{name}'}

"""
# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted Bank Note with the confidence
@app.get('/predict')
def predict_note():

    col_index = len(data_train.columns)
    row_index =np.where(data_train['SK_ID_CURR'] == id)
    column_value_id = data_train.iloc[row_index[0][0], 1:col_index]
    df_column_value_id = column_value_id.to_frame().T
    prediction=pipeline_process.predict_proba(df_column_value_id)[:,1]
    print(prediction)
    return prediction

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
"""    
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

#uvicorn app:app --reload