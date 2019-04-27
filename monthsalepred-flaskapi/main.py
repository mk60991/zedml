
from flask import Flask, request, jsonify
from sklearn.externals import joblib
import traceback
import pandas as pd
import numpy as np
from flask import request
from datetime import datetime
from flask_cors import CORS, cross_origin
from wtforms import TextField,TextAreaField, SubmitField
from wtforms.validators import Required
 
import sys
import os

import pickle
# Preparing the Classifier
cur_dir = os.path.dirname('__file__')
regressor = pickle.load(open(os.path.join(cur_dir,
			'pkl_objects/model.pkl'), 'rb'))
model_columns = pickle.load(open(os.path.join(cur_dir,
			'pkl_objects/model_columns.pkl'),'rb')) 

# Your API definition
app = Flask(__name__)

#for localhost
#cors = CORS(app, resources={r"/": {"origins": "http://localhost:5000"}})

#@app.route('/', methods=['POST'])
#@cross_origin(origin='localhost',headers=['Content- Type','Authorization'])

#for gcp cloud
cors = CORS(app, resources={r"/": {"origins": "https://jts-board.appspot.com/"}})

@app.route('/', methods=['POST'])
@cross_origin(origin='*',headers=['Content- Type','Authorization'])

def predict():
    #print(request)
    if regressor:
        try:
            json_ = request.json
            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(columns=model_columns, fill_value=0)
            
            prediction = list(regressor.predict(query).astype("int64"))
            #print(prediction)
            
                
            return jsonify({'prediction': str(prediction)})

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    app.run(debug=True)