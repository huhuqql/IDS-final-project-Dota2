from flask import Flask, render_template, request, jsonify, send_from_directory
from sklearn.externals import joblib
import json
import os
import csv
import numpy as np

app = Flask(__name__,static_url_path='')


@app.route('/')
@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route('/home.html')
def home():
    return render_template('home.html')

@app.route('/prediction.html')
def prediction():
    return render_template('prediction.html')

@app.route('/methods.html')
def methods():
    return render_template('methods.html')

@app.route('/battle')
def predict():
    model = joblib.load('LRmodel.pkl') 
    # a = request.args.get('a', 0, type=float)
    # b = request.args.get('b', 0, type=float)
    h1 = request.args.get('hero1', 0, type=int)
    h2 = request.args.get('hero2', 0, type=int)
    h3 = request.args.get('hero3', 0, type=int)
    h4 = request.args.get('hero4', 0, type=int)
    h5 = request.args.get('hero5', 0, type=int)
    h6 = request.args.get('hero6', 0, type=int)
    h7 = request.args.get('hero7', 0, type=int)
    h8 = request.args.get('hero8', 0, type=int)
    h9 = request.args.get('hero9', 0, type=int)
    h10 = request.args.get('hero10', 0, type=int)

    newhero1 = [h1,h2,h3,h4,h5]
    newhero2 = [h6,h7,h8,h9,h10]

    num_half_feats = 120
    newdata1 = np.zeros(num_half_feats)
    newdata2 = np.zeros(num_half_feats)
    for hero1 in newhero1:
        newdata1[hero1-1] = 1.0
    for hero2 in newhero2:
        newdata2[hero2-1] = 1.0
    newdata = np.asarray([newdata1 - newdata2]).reshape(1, num_half_feats)
    predict_label = model.predict(newdata)
    predict_prob = model.predict_proba(newdata)
    prob = predict_prob[0]
    if predict_label[0] == 0.0:
        result = "radiant"
        p = prob[0]
    else:
        result = "dire"
        p = prob[1]
    return json.dumps({'winner':result, 'prob':p})

if __name__ == '__main__':
    app.run(host='0.0.0.0')
