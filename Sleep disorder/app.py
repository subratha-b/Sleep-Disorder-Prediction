from flask import Flask, request, jsonify, render_template, redirect, url_for
import numpy as np
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load everything
scaler = joblib.load('scaler.pkl')
model_disorder = joblib.load('model_disorder.pkl')
features_disorder = joblib.load('features_disorder.pkl')
report_disorder = joblib.load('report_disorder.pkl')

model_insomnia = joblib.load('model_insomnia.pkl')
features_insomnia = joblib.load('features_insomnia.pkl')
report_insomnia = joblib.load('report_insomnia.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = request.form.get('age')
    sleepDuration = request.form.get('sleepDuration')
    qualityOfSleep = request.form.get('qualityOfSleep')
    physicalActivityLevel = request.form.get('physicalActivityLevel')
    stressLevel = request.form.get('stressLevel')
    diastolic = request.form.get('diastolic')
    systolic = request.form.get('systolic')
    heartRate = request.form.get('heartRate')
    dailySteps = request.form.get('dailySteps')

    new_data = np.array([[age, sleepDuration, qualityOfSleep, physicalActivityLevel,
                          stressLevel, diastolic, systolic, heartRate, dailySteps]], dtype=float)
    new_data_scaled = scaler.transform(new_data)

    # First predict disorder
    prediction_disorder = model_disorder.predict(new_data_scaled[:, features_disorder])
    if prediction_disorder[0] == 0:
        return redirect(url_for('result',
                                result='No Sleep Disorder',
                                accuracy=report_disorder['0']['precision'],
                                precision=report_disorder['0']['precision'],
                                recall=report_disorder['0']['recall']))
    else:
        prediction_insomnia = model_insomnia.predict(new_data_scaled[:, features_insomnia])
        if prediction_insomnia[0] == 0:
            return redirect(url_for('result',
                                    result='Sleep Apnea',
                                    accuracy=report_insomnia['0']['precision'],
                                    precision=report_insomnia['0']['precision'],
                                    recall=report_insomnia['0']['recall']))
        else:
            return redirect(url_for('result',
                                    result='Insomnia',
                                    accuracy=report_insomnia['1']['precision'],
                                    precision=report_insomnia['1']['precision'],
                                    recall=report_insomnia['1']['recall']))

@app.route('/result')
def result():
    return render_template('result.html',
                           result=request.args.get('result'),
                           accuracy=request.args.get('accuracy'),
                           precision=request.args.get('precision'),
                           recall=request.args.get('recall'))

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
