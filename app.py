
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
from processing_functions import add_variable, get_client
from pydantic import BaseModel


# 2. Create the app object /  Initialize an instance of FastAPI
app = FastAPI()

#import model 
pickle_in = open("pipeline_bank.pkl","rb")
pipeline_process=pickle.load(pickle_in)

data_train = pd.read_csv('application_train.csv')

add_variable(data_train)

#get_client(id,data_train)
@app.get('/')
def index():
    return {'Welcome to this API for loan advisor'}

@app.get('/list')
def get_list_id():
    return data_train['SK_ID_CURR'].tolist()

#Class which describes a single id
class Item(BaseModel):
    id: int
@app.get('/client')
def get_client(item : Item):
    data = item.dict()
    X = data_train[data_train['SK_ID_CURR'] == data['id']]
    return {'gender': X['CODE_GENDER'].values[0], 'credit': X['AMT_CREDIT'].values[0], 'income_tot': X['AMT_INCOME_TOTAL'].values[0], 'income_per': X['INCOME_PER_PERSON'].values[0]}

@app.post('/predict')
def predict_bank(item : Item):
    data = item.dict()
    row_index = data_train['SK_ID_CURR'].tolist().index(data['id'])
    column_value_id = data_train.iloc[row_index][2:]
    df_column_value_id = column_value_id.to_frame().T
    prediction=pipeline_process.predict_proba(df_column_value_id)[:,1]
    id_score = prediction[0]
    return id_score


# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

#uvicorn app:app --reload
