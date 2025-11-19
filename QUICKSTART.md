# Quick Start Guide

This guide will help you get started with the Concrete Compressive Strength Prediction project in minutes.

## 🚀 5-Minute Setup

### Step 1: Install Dependencies

```bash
# Create and activate virtual environment
python -m venv venv

# On Linux/Mac
source venv/bin/activate

# On Windows
venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### Step 2: Train the Model

```bash
python train.py
```

This will:
- Load the concrete dataset
- Train multiple models with hyperparameter tuning
- Save the best model to `models/best_model.pkl`
- Takes about 2-3 minutes

### Step 3: Start the API Server

```bash
python predict.py
```

### Step 4: Test the API

Open your browser and go to:
```
http://localhost:5000
```

Or test with curl:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "cement": 540.0,
    "slag": 0.0,
    "fly_ash": 0.0,
    "water": 162.0,
    "superplasticizer": 2.5,
    "coarse_aggregate": 1040.0,
    "fine_aggregate": 676.0,
    "age": 28
  }'
```

Expected response:
```json
{
  "predicted_strength_MPa": 79.99,
  "model_used": "XGBRegressor",
  "status": "success"
}
```

---

## 🐳 Docker Quick Start

If you prefer Docker:

### Build and Run

```bash
# Build the image
docker build -t concrete-predictor .

# Train the model (one-time)
docker run -v $(pwd)/models:/app/models concrete-predictor python train.py

# Run the API server
docker run -p 5000:5000 -v $(pwd)/models:/app/models concrete-predictor
```

### Using Docker Compose

```bash
cd deploy
docker-compose up
```

---

## 📊 Exploring the Notebook

To see the full exploratory data analysis:

```bash
jupyter notebook "Final Version - Concrete Compressive Strength.ipynb"
```

The notebook includes:
- Data exploration and visualization
- Statistical analysis
- Feature importance analysis
- Model comparison
- Hyperparameter tuning results

---

## 🧪 Running Tests

Test all API endpoints:

```bash
# Make sure the API is running first (python predict.py)

# Then in a new terminal:
python test_api.py
```

---

## 📝 Example Use Cases

### Use Case 1: Quality Control

Check if a concrete mixture will meet strength requirements:

```python
import requests

mixture = {
    "cement": 400.0,
    "slag": 100.0,
    "fly_ash": 0.0,
    "water": 170.0,
    "superplasticizer": 6.0,
    "coarse_aggregate": 1000.0,
    "fine_aggregate": 750.0,
    "age": 28
}

response = requests.post('http://localhost:5000/predict', json=mixture)
result = response.json()

required_strength = 40.0  # MPa
predicted = result['predicted_strength_MPa']

if predicted >= required_strength:
    print(f"✓ PASS: {predicted:.2f} MPa >= {required_strength} MPa")
else:
    print(f"✗ FAIL: {predicted:.2f} MPa < {required_strength} MPa")
```

### Use Case 2: Mixture Optimization

Find the optimal age for desired strength:

```python
import requests

base_mixture = {
    "cement": 350.0,
    "slag": 50.0,
    "fly_ash": 50.0,
    "water": 180.0,
    "superplasticizer": 5.0,
    "coarse_aggregate": 1000.0,
    "fine_aggregate": 750.0
}

target_strength = 45.0
ages = [3, 7, 14, 28, 56, 90]

print(f"Finding age for {target_strength} MPa strength:\n")

for age in ages:
    mixture = {**base_mixture, "age": age}
    response = requests.post('http://localhost:5000/predict', json=mixture)
    strength = response.json()['predicted_strength_MPa']

    print(f"Age {age:3d} days: {strength:.2f} MPa", end="")

    if strength >= target_strength:
        print(" ← Target reached!")
        break
    else:
        print()
```

### Use Case 3: Batch Testing

Test multiple mixtures at once:

```python
import requests

mixtures = {
    "samples": [
        {
            "cement": 400.0, "slag": 0.0, "fly_ash": 0.0,
            "water": 160.0, "superplasticizer": 5.0,
            "coarse_aggregate": 1050.0, "fine_aggregate": 700.0,
            "age": 28
        },
        {
            "cement": 300.0, "slag": 100.0, "fly_ash": 50.0,
            "water": 180.0, "superplasticizer": 5.0,
            "coarse_aggregate": 1000.0, "fine_aggregate": 750.0,
            "age": 28
        },
        {
            "cement": 500.0, "slag": 0.0, "fly_ash": 100.0,
            "water": 170.0, "superplasticizer": 8.0,
            "coarse_aggregate": 1000.0, "fine_aggregate": 700.0,
            "age": 28
        }
    ]
}

response = requests.post('http://localhost:5000/batch_predict', json=mixtures)
results = response.json()

print("Batch Prediction Results:\n")
for i, strength in enumerate(results['predictions'], 1):
    print(f"Mixture {i}: {strength:.2f} MPa")
```

---

## 🎯 Next Steps

1. **Explore the Data**: Open the Jupyter notebook to understand the dataset
2. **Customize the Model**: Modify `train.py` to experiment with different algorithms
3. **Deploy to Cloud**: Use scripts in `deploy/` folder for cloud deployment
4. **Build a Frontend**: Create a web UI on top of the Flask API
5. **Monitor Performance**: Add logging and metrics collection

---

## ❓ Troubleshooting

**Problem**: ModuleNotFoundError
**Solution**: Make sure you activated the virtual environment and installed requirements

**Problem**: FileNotFoundError: models/best_model.pkl
**Solution**: Run `python train.py` first to create the model

**Problem**: Port 5000 already in use
**Solution**: Change the port in `predict.py` or kill the process using port 5000

**Problem**: Docker build fails
**Solution**: Ensure Docker is running and you have enough disk space

---

## 📚 Additional Resources

- Full documentation: See [README.md](README.md)
- Cloud deployment: See [deploy/README.md](deploy/README.md)
- API reference: Check the docstrings in `predict.py`

---

## 🎉 You're All Set!

You now have a fully functional concrete strength prediction system. Happy predicting!
