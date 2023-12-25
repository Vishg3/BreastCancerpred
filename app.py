import pickle
from flask import Flask, render_template,request,jsonify,app,url_for
import numpy as np 
import pandas as pd 

app=Flask(__name__)

regmodel=pickle.load(open('regmodel.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data = np.array(list(data.values())).reshape(1, -1)
    output=regmodel.predict(new_data)
    output_as_python_int = output[0].item() 
    return jsonify(output_as_python_int)

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=np.array(data).reshape(1, -1)
    print(final_input)
    output=regmodel.predict(final_input)[0]
    return render_template('home.html',prediction_text=f'The predicted diagnosis is {output}')

if __name__=='__main__':
    app.run(debug=True)
