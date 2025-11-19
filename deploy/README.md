# Cloud Deployment Guide

This directory contains deployment scripts and configurations for various cloud platforms.

## Prerequisites

Before deploying, ensure you have:
1. Trained the model by running `python train.py` in the parent directory
2. Installed the respective cloud provider's CLI tools
3. Authenticated with your cloud provider account

---

## AWS Elastic Beanstalk

### Setup
```bash
# Install AWS CLI and EB CLI
pip install awscli awsebcli

# Configure AWS credentials
aws configure
```

### Deploy
```bash
cd deploy
bash aws_deploy.sh
```

### Manual Deployment Steps
```bash
# Initialize (from project root)
eb init -p docker concrete-strength-predictor --region us-east-1

# Create environment and deploy
eb create concrete-predictor-env --instance-type t2.small

# Deploy updates
eb deploy

# Open application
eb open

# View logs
eb logs

# Terminate
eb terminate concrete-predictor-env
```

**Cost:** ~$15-30/month for t2.small instance

---

## Google Cloud Run

### Setup
```bash
# Install gcloud CLI
# https://cloud.google.com/sdk/docs/install

# Initialize and authenticate
gcloud init
gcloud auth login

# Set project
gcloud config set project YOUR_PROJECT_ID
```

### Deploy
```bash
cd deploy
bash gcp_deploy.sh
```

### Manual Deployment Steps
```bash
# Enable APIs
gcloud services enable cloudbuild.googleapis.com run.googleapis.com

# Build and push image
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/concrete-predictor

# Deploy to Cloud Run
gcloud run deploy concrete-predictor \
    --image gcr.io/YOUR_PROJECT_ID/concrete-predictor \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 2Gi

# Get service URL
gcloud run services describe concrete-predictor \
    --platform managed \
    --region us-central1 \
    --format 'value(status.url)'
```

**Cost:** Pay per request, ~$0-5/month for low traffic (includes generous free tier)

---

## Heroku

### Setup
```bash
# Install Heroku CLI
# https://devcenter.heroku.com/articles/heroku-cli

# Login
heroku login
heroku container:login
```

### Deploy
```bash
cd deploy
bash heroku_deploy.sh
```

### Manual Deployment Steps
```bash
# Create app
heroku create your-app-name

# Set to container stack
heroku stack:set container

# Build and push
heroku container:push web

# Release
heroku container:release web

# Open app
heroku open

# View logs
heroku logs --tail
```

**Cost:** $7/month for Hobby Dyno (or free tier with limitations)

---

## Docker Compose (Local or VPS)

For deployment on your own server or VPS:

```bash
cd deploy
docker-compose up -d
```

Access at: `http://your-server-ip:5000`

To stop:
```bash
docker-compose down
```

---

## Testing Deployed Application

Once deployed, test the API:

```bash
# Health check
curl https://your-deployed-url.com/health

# Make prediction
curl -X POST https://your-deployed-url.com/predict \
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

---

## Cost Comparison

| Platform | Monthly Cost | Pros | Cons |
|----------|-------------|------|------|
| **GCP Cloud Run** | $0-5 | Pay per use, auto-scaling, generous free tier | Cold starts |
| **Heroku** | $7+ | Easy setup, managed | Limited free tier |
| **AWS EB** | $15-30 | Full control, AWS ecosystem | More complex, always on |
| **Docker on VPS** | $5-10 | Full control, cheapest | Self-managed |

---

## Troubleshooting

### Common Issues

1. **Build fails due to missing dependencies**
   - Ensure `requirements.txt` is up to date
   - Check Docker build logs

2. **Model not found errors**
   - Make sure to train the model first: `python train.py`
   - Check that `models/` directory exists

3. **Memory issues**
   - Increase container memory allocation
   - Use lighter model (e.g., Random Forest instead of XGBoost)

4. **Port binding issues**
   - Ensure port 5000 is not already in use
   - Check firewall settings on VPS

---

## Security Considerations

For production deployments:

1. **Enable HTTPS** - Use platform SSL/TLS certificates
2. **Add authentication** - Implement API keys or OAuth
3. **Rate limiting** - Prevent API abuse
4. **Input validation** - Already implemented in Flask app
5. **Monitoring** - Set up logging and alerts
6. **Secrets management** - Use environment variables for sensitive data

---

## Support

For issues or questions, please refer to the main README.md or create an issue in the repository.
