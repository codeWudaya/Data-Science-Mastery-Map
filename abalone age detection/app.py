from flask import Flask,request, render_template
import numpy as np
import pandas as pd
import pickle


# load modle
model = pickle.load(open('model.pkl','rb'))

#create app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    # sex, length, diameter, height, wholeWeight, Shuckedweight, Visceraweight, Shellweight
    sex = int(request.form['sex'])
    length = float(request.form['length'])
    diameter = float(request.form['diameter'])
    height = float(request.form['height'])
    wholeWeight = float(request.form['wholeWeight'])
    Shuckedweight = float(request.form['Shuckedweight'])
    Visceraweight = float(request.form['Visceraweight'])
    Shellweight = float(request.form['Shellweight'])

    features = np.array([[sex, length, diameter, height, wholeWeight, Shuckedweight, Visceraweight, Shellweight]])

    age = model.predict(features).reshape(1,-1)[0]
    return render_template('index.html',age = age)

# python main
if __name__ == "__main__":
    app.run(debug=True)

