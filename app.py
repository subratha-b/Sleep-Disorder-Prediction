from flask import Flask, request, jsonify, render_template, redirect, url_for
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load and preprocess dataset
data_path = "C:\\Users\\bsupr\\Downloads\\Sleep disorder\\Sleep disorder\\Sleep_health_and_lifestyle_dataset.csv"
data = pd.read_csv(data_path)
data.dropna(inplace=True)

# Drop unnecessary columns
drop_columns = ['Person ID', 'Occupation', 'Gender', 'BMI Category']
data.drop(columns=drop_columns, inplace=True)

# Features and Targets
X = data.drop(columns=['Sleep Disorder', 'Insomnia'])
y_disorder = data['Sleep Disorder']
y_insomnia = data['Insomnia']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_scaled, y_disorder, test_size=0.3, random_state=42, stratify=y_disorder)
X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(X_scaled, y_insomnia, test_size=0.3, random_state=42, stratify=y_insomnia)

# RFE for Disorder
rfe_d = RFE(LogisticRegression(solver='liblinear'), n_features_to_select=9)
rfe_d.fit(X_train_d, y_train_d)
X_train_d_selected = X_train_d[:, rfe_d.support_]
X_test_d_selected = X_test_d[:, rfe_d.support_]

# Train final model for Disorder
clf_d = LogisticRegression(solver='liblinear')
clf_d.fit(X_train_d_selected, y_train_d)

# Grid Search for Disorder
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
grid_d = GridSearchCV(LogisticRegression(solver='liblinear'), param_grid, cv=5)
grid_d.fit(X_train_d_selected, y_train_d)
best_C_d = grid_d.best_params_['C']

# Disorder Predictions
y_pred_d = clf_d.predict(X_test_d_selected)
accuracy_d = round(accuracy_score(y_test_d, y_pred_d), 2)
report_d = classification_report(y_test_d, y_pred_d, output_dict=True)

# RFE for Insomnia
rfe_i = RFE(LogisticRegression(solver='liblinear'), n_features_to_select=9)
rfe_i.fit(X_train_i, y_train_i)
X_train_i_selected = X_train_i[:, rfe_i.support_]
X_test_i_selected = X_test_i[:, rfe_i.support_]

# Train final model for Insomnia
clf_i = LogisticRegression(solver='liblinear')
clf_i.fit(X_train_i_selected, y_train_i)

# Grid Search for Insomnia
grid_i = GridSearchCV(LogisticRegression(solver='liblinear'), param_grid, cv=5)
grid_i.fit(X_train_i_selected, y_train_i)
best_C_i = grid_i.best_params_['C']

# Insomnia Predictions
y_pred_i = clf_i.predict(X_test_i_selected)
accuracy_i = round(accuracy_score(y_test_i, y_pred_i), 2)
report_i = classification_report(y_test_i, y_pred_i, output_dict=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input
        input_values = [
            float(request.form.get('age')),
            float(request.form.get('sleepDuration')),
            float(request.form.get('qualityOfSleep')),
            float(request.form.get('physicalActivityLevel')),
            float(request.form.get('stressLevel')),
            float(request.form.get('diastolic')),
            float(request.form.get('systolic')),
            float(request.form.get('heartRate')),
            float(request.form.get('dailySteps'))
        ]

        input_array = np.array([input_values])
        input_scaled = scaler.transform(input_array)

        # Predict Disorder
        pred_disorder = clf_d.predict(input_scaled[:, rfe_d.support_])[0]
        if pred_disorder == 0:
            return redirect(url_for('result',
                                    result='No Sleep Disorder',
                                    accuracy=accuracy_d,
                                    precision=round(report_d['0']['precision'], 2),
                                    recall=round(report_d['0']['recall'], 2)))
        else:
            # Predict Insomnia if there is a disorder
            pred_insomnia = clf_i.predict(input_scaled[:, rfe_i.support_])[0]
            if pred_insomnia == 0:
                return redirect(url_for('result',
                                        result='Sleep Apnea',
                                        accuracy=accuracy_i,
                                        precision=round(report_i['0']['precision'], 2),
                                        recall=round(report_i['0']['recall'], 2)))
            else:
                return redirect(url_for('result',
                                        result='Insomnia',
                                        accuracy=accuracy_i,
                                        precision=round(report_i['1']['precision'], 2),
                                        recall=round(report_i['1']['recall'], 2)))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/result')
def result():
    return render_template('result.html',
                           result=request.args.get('result'),
                           accuracy=request.args.get('accuracy'),
                           precision=request.args.get('precision'),
                           recall=request.args.get('recall'))

if __name__ == '__main__':
    app.run(debug=True)
