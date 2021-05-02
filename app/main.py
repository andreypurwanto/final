from flask import Flask,request, url_for, redirect, render_template, jsonify
from pycaret.regression import *
import pandas as pd
import pickle
import numpy as np
from config import *

# print(PORT)

app = Flask(__name__)

# model = load_model('deployment_28042020')
cols = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
dict_test ={
    'age':'', 
    'sex':'', 
    'bmi':'', 
    'children':'', 
    'smoker':'', 
    'region':'',
    'tes':'',
    'tes2':'',
}
@app.route('/')
def home():
    # print(request.form.values())
    return render_template("home.html")

@app.route('/predict',methods=['POST'])
def predict():
    
    # int_features = [x for x in request.form.values()]
    print(request.form)
    # final = np.array(int_features)
    # data_unseen = pd.DataFrame([final], columns = cols)
    # prediction = predict_model(model, data=data_unseen, round = 0)
    # prediction = int(prediction.Label[0])

    # print(render_template('home.html',pred='Expected Bill will be {}'.format(prediction)))
    return render_template('home.html',pred=request.form)
    # return request.form

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = predict_model(model, data=data_unseen)
    output = prediction.Label[0]
    return jsonify(output)

# if __name__ == '__main__':
#     app.run(host="0.0.0.0", port=PORT, debug=DEBUG_MODE)
