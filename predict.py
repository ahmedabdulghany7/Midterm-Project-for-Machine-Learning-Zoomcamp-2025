#!/usr/bin/env python
"""
Concrete Compressive Strength Prediction - Flask API
This script provides a REST API for making predictions.
"""

from flask import Flask, request, jsonify, render_template_string
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load model artifacts
MODEL_DIR = 'models'
model = None
scaler = None
feature_names = None


def load_artifacts():
    """Load model, scaler, and feature names."""
    global model, scaler, feature_names

    model_path = os.path.join(MODEL_DIR, 'best_model.pkl')
    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
    features_path = os.path.join(MODEL_DIR, 'feature_names.pkl')

    # Check if files exist
    if not all(os.path.exists(p) for p in [model_path, scaler_path, features_path]):
        raise FileNotFoundError(
            "Model artifacts not found! Please run 'python train.py' first."
        )

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    with open(features_path, 'rb') as f:
        feature_names = pickle.load(f)

    print("✓ Model artifacts loaded successfully!")
    print(f"  Model type: {type(model).__name__}")
    print(f"  Features: {len(feature_names)}")


# HTML template for web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Concrete Strength Predictor</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .form-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            color: #555;
            font-weight: bold;
        }
        input[type="number"] {
            width: 100%;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-sizing: border-box;
        }
        button {
            width: 100%;
            padding: 12px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            margin-top: 10px;
        }
        button:hover {
            background-color: #45a049;
        }
        #result {
            margin-top: 20px;
            padding: 15px;
            border-radius: 4px;
            display: none;
        }
        .success {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
        }
        .error {
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
        }
        .info {
            background-color: #d1ecf1;
            border: 1px solid #bee5eb;
            color: #0c5460;
            padding: 10px;
            border-radius: 4px;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🏗️ Concrete Compressive Strength Predictor</h1>

        <div class="info">
            <strong>Instructions:</strong> Enter the concrete mixture components and age to predict compressive strength.
        </div>

        <form id="predictionForm">
            <div class="form-group">
                <label for="cement">Cement (kg/m³):</label>
                <input type="number" id="cement" name="cement" step="0.1" required
                       placeholder="e.g., 540.0" value="540.0">
            </div>

            <div class="form-group">
                <label for="slag">Blast Furnace Slag (kg/m³):</label>
                <input type="number" id="slag" name="slag" step="0.1" required
                       placeholder="e.g., 0.0" value="0.0">
            </div>

            <div class="form-group">
                <label for="fly_ash">Fly Ash (kg/m³):</label>
                <input type="number" id="fly_ash" name="fly_ash" step="0.1" required
                       placeholder="e.g., 0.0" value="0.0">
            </div>

            <div class="form-group">
                <label for="water">Water (kg/m³):</label>
                <input type="number" id="water" name="water" step="0.1" required
                       placeholder="e.g., 162.0" value="162.0">
            </div>

            <div class="form-group">
                <label for="superplasticizer">Superplasticizer (kg/m³):</label>
                <input type="number" id="superplasticizer" name="superplasticizer" step="0.1" required
                       placeholder="e.g., 2.5" value="2.5">
            </div>

            <div class="form-group">
                <label for="coarse_aggregate">Coarse Aggregate (kg/m³):</label>
                <input type="number" id="coarse_aggregate" name="coarse_aggregate" step="0.1" required
                       placeholder="e.g., 1040.0" value="1040.0">
            </div>

            <div class="form-group">
                <label for="fine_aggregate">Fine Aggregate (kg/m³):</label>
                <input type="number" id="fine_aggregate" name="fine_aggregate" step="0.1" required
                       placeholder="e.g., 676.0" value="676.0">
            </div>

            <div class="form-group">
                <label for="age">Age (days):</label>
                <input type="number" id="age" name="age" step="1" required
                       placeholder="e.g., 28" value="28">
            </div>

            <button type="submit">Predict Strength</button>
        </form>

        <div id="result"></div>
    </div>

    <script>
        document.getElementById('predictionForm').addEventListener('submit', async (e) => {
            e.preventDefault();

            const formData = {
                cement: parseFloat(document.getElementById('cement').value),
                slag: parseFloat(document.getElementById('slag').value),
                fly_ash: parseFloat(document.getElementById('fly_ash').value),
                water: parseFloat(document.getElementById('water').value),
                superplasticizer: parseFloat(document.getElementById('superplasticizer').value),
                coarse_aggregate: parseFloat(document.getElementById('coarse_aggregate').value),
                fine_aggregate: parseFloat(document.getElementById('fine_aggregate').value),
                age: parseInt(document.getElementById('age').value)
            };

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(formData)
                });

                const data = await response.json();
                const resultDiv = document.getElementById('result');

                if (data.status === 'success') {
                    resultDiv.className = 'success';
                    resultDiv.innerHTML = `
                        <h3>Prediction Result:</h3>
                        <p><strong>Predicted Compressive Strength:</strong> ${data.predicted_strength_MPa.toFixed(2)} MPa</p>
                        <p><strong>Model Used:</strong> ${data.model_used}</p>
                    `;
                } else {
                    resultDiv.className = 'error';
                    resultDiv.innerHTML = `<strong>Error:</strong> ${data.message}`;
                }

                resultDiv.style.display = 'block';
            } catch (error) {
                const resultDiv = document.getElementById('result');
                resultDiv.className = 'error';
                resultDiv.innerHTML = `<strong>Error:</strong> Failed to connect to server`;
                resultDiv.style.display = 'block';
            }
        });
    </script>
</body>
</html>
"""


@app.route('/')
def home():
    """Serve the web interface."""
    return render_template_string(HTML_TEMPLATE)


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint.

    Expected JSON format:
    {
        "cement": 540.0,
        "slag": 0.0,
        "fly_ash": 0.0,
        "water": 162.0,
        "superplasticizer": 2.5,
        "coarse_aggregate": 1040.0,
        "fine_aggregate": 676.0,
        "age": 28
    }
    """
    try:
        # Get JSON data
        data = request.get_json()

        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No data provided'
            }), 400

        # Extract features in correct order
        feature_mapping = {
            'cement': 'Cement (component 1)(kg in a m^3 mixture)',
            'slag': 'Blast Furnace Slag (component 2)(kg in a m^3 mixture)',
            'fly_ash': 'Fly Ash (component 3)(kg in a m^3 mixture)',
            'water': 'Water  (component 4)(kg in a m^3 mixture)',
            'superplasticizer': 'Superplasticizer (component 5)(kg in a m^3 mixture)',
            'coarse_aggregate': 'Coarse Aggregate  (component 6)(kg in a m^3 mixture)',
            'fine_aggregate': 'Fine Aggregate (component 7)(kg in a m^3 mixture)',
            'age': 'Age (day)'
        }

        # Build feature array
        features = []
        for short_name, full_name in feature_mapping.items():
            if short_name not in data:
                return jsonify({
                    'status': 'error',
                    'message': f'Missing required field: {short_name}'
                }), 400
            features.append(float(data[short_name]))

        # Convert to numpy array and reshape
        features_array = np.array(features).reshape(1, -1)

        # Scale features
        features_scaled = scaler.transform(features_array)

        # Make prediction
        prediction = model.predict(features_scaled)[0]

        # Return result
        return jsonify({
            'status': 'success',
            'predicted_strength_MPa': float(prediction),
            'model_used': type(model).__name__,
            'input_features': data
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction endpoint.

    Expected JSON format:
    {
        "samples": [
            {"cement": 540.0, "slag": 0.0, ...},
            {"cement": 300.0, "slag": 100.0, ...},
            ...
        ]
    }
    """
    try:
        data = request.get_json()

        if not data or 'samples' not in data:
            return jsonify({
                'status': 'error',
                'message': 'No samples provided'
            }), 400

        samples = data['samples']
        predictions = []

        for sample in samples:
            # Use the predict endpoint logic
            feature_mapping = {
                'cement': 'Cement (component 1)(kg in a m^3 mixture)',
                'slag': 'Blast Furnace Slag (component 2)(kg in a m^3 mixture)',
                'fly_ash': 'Fly Ash (component 3)(kg in a m^3 mixture)',
                'water': 'Water  (component 4)(kg in a m^3 mixture)',
                'superplasticizer': 'Superplasticizer (component 5)(kg in a m^3 mixture)',
                'coarse_aggregate': 'Coarse Aggregate  (component 6)(kg in a m^3 mixture)',
                'fine_aggregate': 'Fine Aggregate (component 7)(kg in a m^3 mixture)',
                'age': 'Age (day)'
            }

            features = [float(sample[short_name]) for short_name in feature_mapping.keys()]
            features_array = np.array(features).reshape(1, -1)
            features_scaled = scaler.transform(features_array)
            prediction = model.predict(features_scaled)[0]
            predictions.append(float(prediction))

        return jsonify({
            'status': 'success',
            'predictions': predictions,
            'count': len(predictions)
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


if __name__ == '__main__':
    print("="*60)
    print("CONCRETE STRENGTH PREDICTION API")
    print("="*60)

    # Load model artifacts
    try:
        load_artifacts()
    except Exception as e:
        print(f"\n❌ Error loading model artifacts: {e}")
        print("\nPlease run 'python train.py' first to train the model.")
        exit(1)

    print("\n" + "="*60)
    print("Starting Flask server...")
    print("="*60)
    print("\nEndpoints:")
    print("  • Web Interface: http://localhost:5000")
    print("  • API Predict: POST http://localhost:5000/predict")
    print("  • Batch Predict: POST http://localhost:5000/batch_predict")
    print("  • Health Check: GET http://localhost:5000/health")
    print("\n" + "="*60)

    app.run(host='0.0.0.0', port=5000, debug=False)
