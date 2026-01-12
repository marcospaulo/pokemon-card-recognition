#!/bin/bash
# Launch EC2 instance for Hailo compilation

set -e

# Configuration
INSTANCE_TYPE="m5.2xlarge"  # 8 vCPUs, 32 GB RAM (~$0.38/hour)
AMI_ID="ami-0c7217cdde317cfec"  # Ubuntu 22.04 LTS in us-east-2
KEY_NAME="${1:-your-key-pair}"  # Pass your key pair name as argument
SECURITY_GROUP="${2:-default}"  # Pass your security group ID
REGION="us-east-2"

echo "======================================"
echo "Launching Hailo Compilation EC2"
echo "======================================"
echo "Instance: $INSTANCE_TYPE"
echo "Region: $REGION"
echo "Key: $KEY_NAME"
echo ""

# Launch instance
echo "Launching EC2 instance..."
INSTANCE_ID=$(aws ec2 run-instances \
    --region $REGION \
    --image-id $AMI_ID \
    --instance-type $INSTANCE_TYPE \
    --key-name $KEY_NAME \
    --security-group-ids $SECURITY_GROUP \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=hailo-compiler},{Key=Project,Value=pokemon-card-recognition}]' \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "✅ Instance launched: $INSTANCE_ID"
echo ""

# Wait for instance to be running
echo "Waiting for instance to be running..."
aws ec2 wait instance-running --region $REGION --instance-ids $INSTANCE_ID

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
    --region $REGION \
    --instance-ids $INSTANCE_ID \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo ""
echo "======================================"
echo "✅ EC2 INSTANCE READY"
echo "======================================"
echo "Instance ID: $INSTANCE_ID"
echo "Public IP: $PUBLIC_IP"
echo "SSH Command: ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@${PUBLIC_IP}"
echo ""
echo "Next steps:"
echo "  1. Wait 30 seconds for SSH to be ready"
echo "  2. SSH to instance"
echo "  3. Run: curl -o setup.sh https://raw.githubusercontent.com/.../setup_hailo_ec2.sh && bash setup.sh"
echo "  4. Download Hailo DFC from hailo.ai"
echo "  5. Copy ONNX model: scp models/onnx/pokemon_student_convnext_tiny.onnx ubuntu@${PUBLIC_IP}:~/"
echo ""
echo "Cost: ~$0.38/hour"
echo "======================================"

# Save instance info
cat > hailo_ec2_info.txt <<EOF
Instance ID: $INSTANCE_ID
Public IP: $PUBLIC_IP
Region: $REGION
Launched: $(date)
SSH: ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@${PUBLIC_IP}
EOF

echo "Instance info saved to: hailo_ec2_info.txt"
