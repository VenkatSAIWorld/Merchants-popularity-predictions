# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 19:28:25 2021

@author: Venkat'sAIWorld
"""

from flask import Flask, render_template, request
import pickle
import numpy as np

app=Flask(__name__)
model=pickle.load(open('rf_mpp.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')
    
@app.route('/predict', methods=['POST'])
def predict():
    values=[int(X) for X in request.form.values()]
    final_features=[np.array(values)]
    prediction=model.predict(final_features)
    output= prediction[0]
    
    return render_template('index.html', prediction_text='output is : {}'.format(output))


if __name__== '__main__':
    app.run(debug=True)