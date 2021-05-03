from flask import Flask,request, url_for, redirect, render_template, jsonify
# from pycaret.regression import *
import pandas as pd
import pickle
import numpy as np
from config import *
import joblib
# print(PORT)
import os
import pickle
app = Flask(__name__)

xgb = pickle.load(open('xgb3.pkl', 'rb'))
order = ['Product_Info_1',
 'Product_Info_3',
 'Product_Info_4',
 'Product_Info_5',
 'Product_Info_6',
 'Product_Info_7',
 'Ins_Age',
 'Ht',
 'Wt',
 'BMI',
 'Employment_Info_1',
 'Employment_Info_2',
 'Employment_Info_3',
 'Employment_Info_4',
 'Employment_Info_5',
 'Employment_Info_6',
 'InsuredInfo_1',
 'InsuredInfo_2',
 'InsuredInfo_3',
 'InsuredInfo_4',
 'InsuredInfo_5',
 'InsuredInfo_6',
 'InsuredInfo_7',
 'Insurance_History_1',
 'Insurance_History_2',
 'Insurance_History_3',
 'Insurance_History_4',
 'Insurance_History_5',
 'Insurance_History_7',
 'Insurance_History_8',
 'Insurance_History_9',
 'Family_Hist_1',
 'Family_Hist_2',
 'Family_Hist_3',
 'Family_Hist_4',
 'Family_Hist_5',
 'Medical_History_1',
 'Medical_History_2',
 'Medical_History_3',
 'Medical_History_4',
 'Medical_History_5',
 'Medical_History_6',
 'Medical_History_7',
 'Medical_History_8',
 'Medical_History_9',
 'Medical_History_11',
 'Medical_History_12',
 'Medical_History_13',
 'Medical_History_14',
 'Medical_History_16',
 'Medical_History_17',
 'Medical_History_18',
 'Medical_History_19',
 'Medical_History_20',
 'Medical_History_21',
 'Medical_History_22',
 'Medical_History_23',
 'Medical_History_25',
 'Medical_History_26',
 'Medical_History_27',
 'Medical_History_28',
 'Medical_History_29',
 'Medical_History_30',
 'Medical_History_31',
 'Medical_History_33',
 'Medical_History_34',
 'Medical_History_35',
 'Medical_History_36',
 'Medical_History_37',
 'Medical_History_38',
 'Medical_History_39',
 'Medical_History_40',
 'Medical_History_41',
 'Medical_Sum',
 'Product_Info_2_A2',
 'Product_Info_2_A3',
 'Product_Info_2_A4',
 'Product_Info_2_A5',
 'Product_Info_2_A6',
 'Product_Info_2_A7',
 'Product_Info_2_A8',
 'Product_Info_2_B1',
 'Product_Info_2_B2',
 'Product_Info_2_C1',
 'Product_Info_2_C2',
 'Product_Info_2_C3',
 'Product_Info_2_C4',
 'Product_Info_2_D1',
 'Product_Info_2_D2',
 'Product_Info_2_D3',
 'Product_Info_2_D4',
 'Product_Info_2_E1']

drop = ['Medical_History_10', 'Medical_History_15', 'Medical_History_24', 'Medical_History_32']
cat = ['D3', 'A1', 'E1', 'D4', 'D2', 'A8', 'A2', 'D1', 'A7', 'A6', 'A3', 'A5', 'C4', 'C1', 'B2', 'C3', 'C2', 'A4', 'B1']
float_col = ['Product_Info_4',
 'Ins_Age',
 'Ht',
 'Wt',
 'BMI',
 'Employment_Info_1',
 'Employment_Info_4',
 'Employment_Info_6',
 'Insurance_History_5',
 'Family_Hist_2',
 'Family_Hist_3',
 'Family_Hist_4',
 'Family_Hist_5',
 'Medical_History_1',
 'Medical_History_10',
 'Medical_History_15',
 'Medical_History_24',
 'Medical_History_32']
int_col = ['Id',
 'Product_Info_1',
 'Product_Info_3',
 'Product_Info_5',
 'Product_Info_6',
 'Product_Info_7',
 'Employment_Info_2',
 'Employment_Info_3',
 'Employment_Info_5',
 'InsuredInfo_1',
 'InsuredInfo_2',
 'InsuredInfo_3',
 'InsuredInfo_4',
 'InsuredInfo_5',
 'InsuredInfo_6',
 'InsuredInfo_7',
 'Insurance_History_1',
 'Insurance_History_2',
 'Insurance_History_3',
 'Insurance_History_4',
 'Insurance_History_7',
 'Insurance_History_8',
 'Insurance_History_9',
 'Family_Hist_1',
 'Medical_History_2',
 'Medical_History_3',
 'Medical_History_4',
 'Medical_History_5',
 'Medical_History_6',
 'Medical_History_7',
 'Medical_History_8',
 'Medical_History_9',
 'Medical_History_11',
 'Medical_History_12',
 'Medical_History_13',
 'Medical_History_14',
 'Medical_History_16',
 'Medical_History_17',
 'Medical_History_18',
 'Medical_History_19',
 'Medical_History_20',
 'Medical_History_21',
 'Medical_History_22',
 'Medical_History_23',
 'Medical_History_25',
 'Medical_History_26',
 'Medical_History_27',
 'Medical_History_28',
 'Medical_History_29',
 'Medical_History_30',
 'Medical_History_31',
 'Medical_History_33',
 'Medical_History_34',
 'Medical_History_35',
 'Medical_History_36',
 'Medical_History_37',
 'Medical_History_38',
 'Medical_History_39',
 'Medical_History_40',
 'Medical_History_41',
 'Medical_Keyword_1',
 'Medical_Keyword_2',
 'Medical_Keyword_3',
 'Medical_Keyword_4',
 'Medical_Keyword_5',
 'Medical_Keyword_6',
 'Medical_Keyword_7',
 'Medical_Keyword_8',
 'Medical_Keyword_9',
 'Medical_Keyword_10',
 'Medical_Keyword_11',
 'Medical_Keyword_12',
 'Medical_Keyword_13',
 'Medical_Keyword_14',
 'Medical_Keyword_15',
 'Medical_Keyword_16',
 'Medical_Keyword_17',
 'Medical_Keyword_18',
 'Medical_Keyword_19',
 'Medical_Keyword_20',
 'Medical_Keyword_21',
 'Medical_Keyword_22',
 'Medical_Keyword_23',
 'Medical_Keyword_24',
 'Medical_Keyword_25',
 'Medical_Keyword_26',
 'Medical_Keyword_27',
 'Medical_Keyword_28',
 'Medical_Keyword_29',
 'Medical_Keyword_30',
 'Medical_Keyword_31',
 'Medical_Keyword_32',
 'Medical_Keyword_33',
 'Medical_Keyword_34',
 'Medical_Keyword_35',
 'Medical_Keyword_36',
 'Medical_Keyword_37',
 'Medical_Keyword_38',
 'Medical_Keyword_39',
 'Medical_Keyword_40',
 'Medical_Keyword_41',
 'Medical_Keyword_42',
 'Medical_Keyword_43',
 'Medical_Keyword_44',
 'Medical_Keyword_45',
 'Medical_Keyword_46',
 'Medical_Keyword_47',
 'Medical_Keyword_48',
 'Response']
@app.route('/')
def home():
    # print(request.form.values())
    return render_template("home.html")

@app.route('/predict',methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        try:
            df_input = pd.read_csv('default_value.csv', index_col=0)
            # df_input = pd.read_csv('train_head.csv', index_col=0)
            df_input = df_input.head()
            result = request.form
            for key in request.form:
                if key in df_input.columns:
                    if key in int_col:
                        if result[key] != '':
                            df_input[key] = result[key]
                            df_input[key] = df_input[key].astype(np.int64) 
                    if key in float_col:
                        if result[key] != '':
                            df_input[key] = result[key]
                            df_input[key] = df_input[key].astype(np.float64)
            medical = []
            for item in df_input.columns:
                if 'Medical_Keyword' in item:
                    medical.append(item)
            df_input["Medical_Sum"] = df_input.apply(lambda row: row[medical].sum(),axis=1)
            df_input= df_input.drop(columns = medical)
            for item in cat:
                if df_input.Product_Info_2.loc[0] == item:
                    df_input['Product_Info_2'+str('_')+item] = 1
                else:
                    df_input['Product_Info_2'+str('_')+item] = 0
            df_input= df_input.drop(columns = ['Product_Info_2'])
            try:
                result_pred = xgb.predict(df_input[order])
            except (Exception) as e:
                result_pred = e
            # df_input[order].to_csv('test.csv')
            return (render_template('home.html',pred='Predicted Customer : {}'.format(result_pred)))
        except (Exception) as e:
            return (render_template('home.html',pred='Error Status : {}'.format(e)))
    else:
        return render_template("home.html")

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = predict_model(model, data=data_unseen)
    output = prediction.Label[0]
    return jsonify(output)

# if __name__ == '__main__':
#     app.run(host="0.0.0.0", port=PORT, debug=DEBUG_MODE)
