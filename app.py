import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from analysis import analyzer

analyzer=analyzer()
app = Flask(__name__,template_folder='templates')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/',methods=['POST'])
def predict():

    text = request.form['Email id']
    # print(jsonify(text))
    output = analyzer.output(text)
    if output == 0:
        return render_template('negative.html')
        # return  jsonify(text)
    if output ==1:
        return render_template('positive.html')
        # return  jsonify(text)



if __name__ == "__main__":
    app.run(debug=True)