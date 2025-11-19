#!/bin/bash

# Heroku Deployment Script
# This script deploys the application to Heroku

echo "=========================================="
echo "Heroku Deployment"
echo "=========================================="

# Check if Heroku CLI is installed
if ! command -v heroku &> /dev/null; then
    echo "❌ Heroku CLI not found. Please install it first:"
    echo "   https://devcenter.heroku.com/articles/heroku-cli"
    exit 1
fi

# Configuration
APP_NAME="concrete-strength-predictor-$(date +%s)"

echo ""
echo "App Name: $APP_NAME"
echo ""

# Login to Heroku
echo "🔐 Logging in to Heroku..."
heroku login

# Login to Heroku Container Registry
echo ""
echo "🔐 Logging in to Heroku Container Registry..."
heroku container:login

# Create Heroku app
echo ""
echo "📦 Creating Heroku app..."
heroku create $APP_NAME

# Set stack to container
heroku stack:set container -a $APP_NAME

# Build and push container
echo ""
echo "🔨 Building and pushing container..."
cd ..
heroku container:push web -a $APP_NAME

# Release the container
echo ""
echo "🚀 Releasing container..."
heroku container:release web -a $APP_NAME

# Open the app
echo ""
echo "=========================================="
echo "✅ Deployment Complete!"
echo "=========================================="
echo ""
echo "Your app URL:"
heroku info -a $APP_NAME | grep "Web URL"
echo ""
echo "To open in browser:"
echo "  heroku open -a $APP_NAME"
echo ""
echo "To view logs:"
echo "  heroku logs --tail -a $APP_NAME"
echo ""
echo "To delete app:"
echo "  heroku apps:destroy $APP_NAME"
echo ""
