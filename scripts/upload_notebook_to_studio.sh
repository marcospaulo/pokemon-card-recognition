#!/bin/bash
# Upload validation notebook to S3 for SageMaker Studio access

BUCKET="pokemon-card-training-us-east-2"
NOTEBOOK_PATH="../notebooks/validate_distilled_model.ipynb"

echo "Uploading validation notebook to S3..."
aws s3 cp "$NOTEBOOK_PATH" "s3://${BUCKET}/notebooks/validate_distilled_model.ipynb"

echo ""
echo "✅ Notebook uploaded!"
echo ""
echo "To access in SageMaker Studio:"
echo "1. Open SageMaker Studio: https://console.aws.amazon.com/sagemaker/home?region=us-east-2#/studio"
echo "2. Go to File → New → Terminal"
echo "3. Run: aws s3 cp s3://${BUCKET}/notebooks/validate_distilled_model.ipynb ."
echo "4. Double-click the notebook to open"
echo ""
echo "Or download directly:"
echo "https://s3.us-east-2.amazonaws.com/${BUCKET}/notebooks/validate_distilled_model.ipynb"
