
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 16:39:04 2020
@author:
"""

from flask import Flask, request
import numpy as np
import pickle
import pandas as pd
import flasgger
from flasgger import Swagger
from processing_functions import add_variable


app=Flask(__name__)
Swagger(app) 
data_train = pd.read_csv('application_train.csv')
pickle_in = open("pipeline_bank.pkl","rb")
classifier=pickle.load(pickle_in)

add_variable(data_train)

@app.route('/predict',methods=["POST"])
def predict_note():
    
    """Let's find the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: id
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """
    id = request.args.get("id")

    col_index = len(data_train.columns)
    row_index =np.where(data_train['SK_ID_CURR'] == id)
    column_value_id = data_train.iloc[row_index[0][0], 1:col_index]
    df_column_value_id = column_value_id.to_frame().T
    prediction=classifier.predict_proba(df_column_value_id)[:,1]

    #"The score is "+str(round(prediction, 4))+" for the client "+str(id)
    return print(prediction)


if __name__=='__main__':
    app.run(debug=True)
    #app.run(host='0.0.0.0',port=8000)
    