#!/bin/bash

# AWS Elastic Beanstalk Deployment Script
# This script deploys the application to AWS Elastic Beanstalk

echo "=========================================="
echo "AWS Elastic Beanstalk Deployment"
echo "=========================================="

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI not found. Please install it first:"
    echo "   pip install awscli"
    exit 1
fi

# Check if EB CLI is installed
if ! command -v eb &> /dev/null; then
    echo "❌ EB CLI not found. Please install it first:"
    echo "   pip install awsebcli"
    exit 1
fi

# Configuration
APP_NAME="concrete-strength-predictor"
ENV_NAME="concrete-predictor-env"
REGION="us-east-1"

echo ""
echo "Application: $APP_NAME"
echo "Environment: $ENV_NAME"
echo "Region: $REGION"
echo ""

# Initialize Elastic Beanstalk (if not already initialized)
if [ ! -d ".elasticbeanstalk" ]; then
    echo "📦 Initializing Elastic Beanstalk..."
    eb init -p docker-20.10.7 $APP_NAME --region $REGION
else
    echo "✓ Elastic Beanstalk already initialized"
fi

# Create environment (if it doesn't exist)
echo ""
echo "🚀 Creating/Updating environment..."
if eb list | grep -q $ENV_NAME; then
    echo "Environment exists, deploying update..."
    eb deploy $ENV_NAME
else
    echo "Creating new environment..."
    eb create $ENV_NAME --instance-type t2.small
fi

echo ""
echo "=========================================="
echo "✅ Deployment Complete!"
echo "=========================================="
echo ""
echo "To view your application:"
echo "  eb open $ENV_NAME"
echo ""
echo "To view logs:"
echo "  eb logs $ENV_NAME"
echo ""
echo "To check status:"
echo "  eb status $ENV_NAME"
echo ""
echo "To terminate (cleanup):"
echo "  eb terminate $ENV_NAME"
echo ""
