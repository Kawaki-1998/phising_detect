#!/bin/bash

echo "This script will help you set up the required secrets for your GitHub repository."
echo "Please make sure you have the following information ready:"
echo "1. Render API Key"
echo "2. Render Service ID"
echo "3. Snyk Token (for security scanning)"
echo "4. Slack Webhook URL (for deployment notifications)"
echo

# Check if gh CLI is installed
if ! command -v gh &> /dev/null; then
    echo "GitHub CLI (gh) is not installed. Please install it first:"
    echo "https://cli.github.com/"
    exit 1
fi

# Check if logged in to GitHub
if ! gh auth status &> /dev/null; then
    echo "Please login to GitHub first:"
    gh auth login
fi

# Get repository name
repo=$(gh repo view --json nameWithOwner -q .nameWithOwner)

echo "Setting up secrets for repository: $repo"
echo

# Render API Key
echo "Enter your Render API Key:"
read -s render_key
echo
gh secret set RENDER_API_KEY -b"$render_key"

# Render Service ID
echo "Enter your Render Service ID:"
read render_service_id
gh secret set RENDER_SERVICE_ID -b"$render_service_id"

# Snyk Token
echo "Enter your Snyk Token (press Enter to skip):"
read -s snyk_token
if [ ! -z "$snyk_token" ]; then
    gh secret set SNYK_TOKEN -b"$snyk_token"
fi

# Slack Webhook
echo "Enter your Slack Webhook URL (press Enter to skip):"
read -s slack_webhook
if [ ! -z "$slack_webhook" ]; then
    gh secret set SLACK_WEBHOOK -b"$slack_webhook"
fi

echo
echo "Secrets have been set up successfully!"
echo "You can now push your code to GitHub and the CI/CD pipeline will be triggered." 