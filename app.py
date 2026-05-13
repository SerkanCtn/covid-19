from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
import pickle
import os

app = Flask(__name__)

# Load model and scaler
model = tf.keras.models.load_model('covid_model.keras')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # EXACT ORDER REQUIRED BY THE MODEL:
        # 0: USMER, 1: MEDICAL_UNIT, 2: SEX, 3: PATIENT_TYPE, 4: INTUBED, 
        # 5: PNEUMONIA, 6: AGE, 7: PREGNANT, 8: DIABETES, 9: COPD, 
        # 10: ASTHMA, 11: INMSUPR, 12: HIPERTENSION, 13: OTHER_DISEASE, 14: CARDIOVASCULAR, 
        # 15: OBESITY, 16: RENAL_CHRONIC, 17: TOBACCO, 18: CLASIFFICATION_FINAL, 19: ICU
        
        features = [
            int(data.get('usmer', 2)),
            int(data.get('medical_unit', 1)),
            int(data.get('sex', 1)),
            int(data.get('patient_type', 1)),
            int(data.get('intubed', 2)),
            int(data.get('pneumonia', 2)),
            int(data.get('age', 30)),
            int(data.get('pregnant', 2)),
            int(data.get('diabetes', 2)),
            int(data.get('copd', 2)),
            int(data.get('asthma', 2)),
            int(data.get('inmsupr', 2)),
            int(data.get('hipertension', 2)),
            int(data.get('other_disease', 2)),
            int(data.get('cardiovascular', 2)),
            int(data.get('obesity', 2)),
            int(data.get('renal_chronic', 2)),
            int(data.get('tobacco', 2)),
            int(data.get('classification', 3)),
            int(data.get('icu', 2))
        ]
        
        # Scale features
        features_scaled = scaler.transform([features])
        
        # Predict
        prediction = model.predict(features_scaled)[0][0]
        
        return jsonify({
            'success': True,
            'prediction': float(prediction),
            'risk_level': 'High' if prediction > 0.5 else 'Low'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
