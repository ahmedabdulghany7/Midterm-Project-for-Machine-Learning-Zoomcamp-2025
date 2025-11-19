# Concrete Compressive Strength Prediction

## Problem Description

Concrete is one of the most important materials in civil engineering. The compressive strength of concrete is a highly nonlinear function of age and ingredients. This project aims to predict the **concrete compressive strength (MPa)** based on the mixture composition and curing age.

### Business Context
Accurately predicting concrete strength is crucial for:
- **Quality control** in construction projects
- **Cost optimization** by fine-tuning mixture ratios
- **Safety assurance** in structural engineering
- **Time efficiency** by reducing the need for physical testing (which typically requires 28 days of curing)

### Dataset Information
The dataset contains **1030 samples** with **8 input features** and **1 target variable**:

**Input Features:**
1. **Cement** (kg/m³) - Primary binding component
2. **Blast Furnace Slag** (kg/m³) - Industrial byproduct used as cement replacement
3. **Fly Ash** (kg/m³) - Coal combustion byproduct
4. **Water** (kg/m³) - Required for hydration
5. **Superplasticizer** (kg/m³) - Chemical admixture for workability
6. **Coarse Aggregate** (kg/m³) - Gravel or crushed stone
7. **Fine Aggregate** (kg/m³) - Sand
8. **Age** (days) - Curing time (1-365 days)

**Target Variable:**
- **Concrete Compressive Strength** (MPa) - Ranges from ~2 MPa to ~82 MPa

### Solution Approach
This project implements a machine learning pipeline to:
1. Perform extensive exploratory data analysis
2. Train multiple regression models (Linear, Tree-based, Neural Networks)
3. Optimize hyperparameters for best performance
4. Deploy the model as a REST API service
5. Containerize the application for reproducibility
6. Provide cloud deployment options

---

## Project Structure

```
.
├── Concrete_Data.xls                          # Original dataset
├── Final Version - Concrete Compressive Strength.ipynb  # Jupyter notebook with EDA & experiments
├── train.py                                   # Training script
├── predict.py                                 # Flask API for predictions
├── models/                                    # Saved models
│   └── best_model.pkl
├── requirements.txt                           # Python dependencies
├── Dockerfile                                 # Docker containerization
├── deploy/                                    # Cloud deployment scripts
│   ├── aws_deploy.sh
│   └── docker-compose.yml
└── README.md                                  # This file
```

---

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Docker (optional, for containerization)

### Local Setup

1. **Clone the repository or download the project files**

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv

   # On Linux/Mac
   source venv/bin/activate

   # On Windows
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model**
   ```bash
   python train.py
   ```
   This will:
   - Load and preprocess the data
   - Train multiple models with hyperparameter tuning
   - Save the best model to `models/best_model.pkl`
   - Display performance metrics

5. **Run the prediction service**
   ```bash
   python predict.py
   ```
   The API will be available at `http://localhost:5000`

---

## Usage

### Making Predictions via API

**Endpoint:** `POST /predict`

**Example Request:**
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

**Example Response:**
```json
{
  "predicted_strength_MPa": 79.99,
  "model_used": "XGBoost",
  "status": "success"
}
```

### Using in Python
```python
import requests

data = {
    "cement": 300,
    "slag": 100,
    "fly_ash": 50,
    "water": 180,
    "superplasticizer": 5,
    "coarse_aggregate": 1000,
    "fine_aggregate": 750,
    "age": 28
}

response = requests.post('http://localhost:5000/predict', json=data)
print(response.json())
```

---

## Docker Deployment

### Build the Docker image
```bash
docker build -t concrete-strength-predictor .
```

### Run the container
```bash
docker run -p 5000:5000 concrete-strength-predictor
```

The API will be accessible at `http://localhost:5000`

---

## Cloud Deployment

### Option 1: AWS Elastic Beanstalk

1. **Install AWS CLI and EB CLI**
   ```bash
   pip install awscli awsebcli
   ```

2. **Initialize and deploy**
   ```bash
   cd deploy
   bash aws_deploy.sh
   ```

3. **Access your application**
   The script will output the URL where your application is deployed.

### Option 2: Google Cloud Run

1. **Build and push to Google Container Registry**
   ```bash
   gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/concrete-predictor
   ```

2. **Deploy to Cloud Run**
   ```bash
   gcloud run deploy concrete-predictor \
     --image gcr.io/YOUR_PROJECT_ID/concrete-predictor \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated
   ```

### Option 3: Heroku

1. **Login to Heroku**
   ```bash
   heroku login
   heroku container:login
   ```

2. **Create app and deploy**
   ```bash
   heroku create concrete-strength-app
   heroku container:push web
   heroku container:release web
   heroku open
   ```

---

## Model Performance

| Model | R² Score | MAE | MSE | Training Time |
|-------|----------|-----|-----|---------------|
| **XGBoost (Tuned)** | 0.923 | 2.91 | 19.82 | ~3s |
| Random Forest (Tuned) | 0.882 | 3.76 | 30.43 | ~2s |
| Decision Tree | 0.849 | 4.12 | 38.76 | ~0.5s |
| KNN | 0.712 | 6.86 | 74.33 | ~0.1s |
| Linear Regression | 0.628 | 7.75 | 95.98 | ~0.1s |
| SVM | 0.648 | 7.57 | 90.71 | ~1s |

**Best Model:** XGBoost with optimized hyperparameters

---

## Key Findings from EDA

1. **Strongest Positive Correlations with Strength:**
   - Cement (r ≈ 0.50)
   - Superplasticizer (r ≈ 0.37)
   - Age (r ≈ 0.33)

2. **Negative Correlation:**
   - Water content has a negative impact on strength (r ≈ -0.29)

3. **Data Quality:**
   - No missing values
   - 25 duplicate records (kept for analysis)
   - Some outliers present but represent valid edge cases

4. **Feature Distributions:**
   - Most features are right-skewed
   - Age has discrete values (common curing periods: 3, 7, 28, 90, 180, 365 days)

---

## Technologies Used

- **Python 3.8+**
- **Machine Learning:** scikit-learn, XGBoost
- **Data Analysis:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **API Framework:** Flask
- **Containerization:** Docker
- **Cloud Platforms:** AWS, GCP, Heroku

---

## Future Improvements

- [ ] Implement neural network models with TensorFlow/PyTorch
- [ ] Add model monitoring and retraining pipeline
- [ ] Create web UI for easier interaction
- [ ] Implement ensemble methods
- [ ] Add A/B testing framework
- [ ] Include uncertainty quantification

---

## License

This project is for educational purposes.

## Author

Ahmed Abdulghany

---

## Acknowledgments

- Dataset source: UCI Machine Learning Repository
- Course: Machine Learning Zoomcamp
