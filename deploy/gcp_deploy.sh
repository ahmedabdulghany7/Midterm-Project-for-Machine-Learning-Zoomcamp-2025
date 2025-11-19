#!/bin/bash

# Google Cloud Platform Deployment Script
# This script deploys the application to Google Cloud Run

echo "=========================================="
echo "Google Cloud Run Deployment"
echo "=========================================="

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI not found. Please install it first:"
    echo "   https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Configuration
PROJECT_ID=$(gcloud config get-value project)
SERVICE_NAME="concrete-predictor"
REGION="us-central1"
IMAGE_NAME="gcr.io/$PROJECT_ID/$SERVICE_NAME"

echo ""
echo "Project ID: $PROJECT_ID"
echo "Service Name: $SERVICE_NAME"
echo "Region: $REGION"
echo "Image: $IMAGE_NAME"
echo ""

# Check if project is set
if [ -z "$PROJECT_ID" ]; then
    echo "❌ No GCP project set. Please run:"
    echo "   gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi

# Enable required APIs
echo "📋 Enabling required Google Cloud APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com

# Build the container image
echo ""
echo "🔨 Building container image..."
cd ..
gcloud builds submit --tag $IMAGE_NAME

# Deploy to Cloud Run
echo ""
echo "🚀 Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
    --image $IMAGE_NAME \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 1 \
    --port 5000

echo ""
echo "=========================================="
echo "✅ Deployment Complete!"
echo "=========================================="
echo ""
echo "Your service URL:"
gcloud run services describe $SERVICE_NAME --platform managed --region $REGION --format 'value(status.url)'
echo ""
echo "To view logs:"
echo "  gcloud run services logs read $SERVICE_NAME --region $REGION"
echo ""
echo "To delete service:"
echo "  gcloud run services delete $SERVICE_NAME --region $REGION"
echo ""
