# Pokemon Card Recognition - Project Access Guide

**Project Owner:** Marcos Paulo (marcospaulo)
**Service Account:** `SageMaker-MarcosAdmin-ExecutionRole`
**Account ID:** 943271038849
**Region:** us-east-2

---

## 🔗 Quick Access Links

### **1. S3 Project Structure**

**Main Project Location:**
```
s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/
```

**AWS Console Link:**
[S3 Project Root](https://s3.console.aws.amazon.com/s3/buckets/pokemon-card-training-us-east-2?region=us-east-2&prefix=project/pokemon-card-recognition/)

**Direct Links to Key Directories:**
- **Models:** [s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/models/](https://s3.console.aws.amazon.com/s3/buckets/pokemon-card-training-us-east-2?prefix=project/pokemon-card-recognition/models/)
- **Analytics:** [s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/analytics/](https://s3.console.aws.amazon.com/s3/buckets/pokemon-card-training-us-east-2?prefix=project/pokemon-card-recognition/analytics/)
- **Profiling:** [s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/profiling/](https://s3.console.aws.amazon.com/s3/buckets/pokemon-card-training-us-east-2?prefix=project/pokemon-card-recognition/profiling/)
- **Data:** [s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/data/](https://s3.console.aws.amazon.com/s3/buckets/pokemon-card-training-us-east-2?prefix=project/pokemon-card-recognition/data/)

---

### **2. SageMaker Console**

**SageMaker Home (us-east-2):**
https://console.aws.amazon.com/sagemaker/home?region=us-east-2

**Model Registry:**
https://console.aws.amazon.com/sagemaker/home?region=us-east-2#/model-packages

**Registered Models:**
- List all models: https://console.aws.amazon.com/sagemaker/home?region=us-east-2#/model-package-groups/pokemon-card-recognition-models

**Training Jobs:**
https://console.aws.amazon.com/sagemaker/home?region=us-east-2#/jobs

**Endpoints (if deployed):**
https://console.aws.amazon.com/sagemaker/home?region=us-east-2#/endpoints

---

### **3. CloudWatch Monitoring**

**CloudWatch Dashboards:**
https://console.aws.amazon.com/cloudwatch/home?region=us-east-2#dashboards:

**Log Groups:**
https://console.aws.amazon.com/cloudwatch/home?region=us-east-2#logsV2:log-groups

**Metrics:**
https://console.aws.amazon.com/cloudwatch/home?region=us-east-2#metricsV2:

---

### **4. IAM Role**

**SageMaker-MarcosAdmin-ExecutionRole:**
https://console.aws.amazon.com/iam/home?region=us-east-2#/roles/SageMaker-MarcosAdmin-ExecutionRole

**Role ARN:**
```
arn:aws:iam::943271038849:role/SageMaker-MarcosAdmin-ExecutionRole
```

---

## 🔐 Service Account Permissions

### **Current Permissions for `SageMaker-MarcosAdmin-ExecutionRole`:**

This role has **FULL ADMIN ACCESS** to everything in the project:

#### **AWS Managed Policies Attached:**
✅ **AmazonSageMakerFullAccess** - Full control over SageMaker resources
✅ **AmazonS3FullAccess** - Full control over all S3 buckets and objects
✅ **IAMFullAccess** - Full IAM permissions
✅ **CloudWatchFullAccess** - Full monitoring and logging access
✅ **AWSCloudFormationFullAccess** - Infrastructure as Code management
✅ **AWSLambda_FullAccess** - Lambda function management
✅ **AWSCodePipeline_FullAccess** - CI/CD pipeline management
✅ **AWSCodeBuildAdminAccess** - Build automation
✅ **AWSServiceCatalogAdminFullAccess** - Service catalog management

#### **Custom Inline Policies:**
✅ **SageMakerCompleteAdminAccess** - Additional SageMaker permissions
✅ **AdditionalAdminPermissions** - Extended admin capabilities

### **What This Role Can Do:**

✅ **Full SageMaker Access:**
- Create, read, update, delete all SageMaker resources
- Launch training jobs
- Deploy models to endpoints
- Manage Model Registry
- Access SageMaker Studio
- Create pipelines and workflows

✅ **Full S3 Access:**
- Read/write all objects in `pokemon-card-training-us-east-2`
- Create/delete buckets
- Manage lifecycle policies
- Configure bucket policies

✅ **Full Monitoring:**
- View CloudWatch logs
- Create dashboards
- Set up alarms
- View metrics

✅ **Full IAM:**
- Create/modify roles
- Attach policies
- Manage permissions

**Bottom Line:** This role has **absolute admin access** to do everything it needs and wants! 🎉

---

## 📊 Project Resources Summary

### **Storage (S3):**
```
Total: ~31.7 GB (53,068 objects)
├─ Models: 5.7 GB (4 variants + calibration data)
├─ Data: 25.2 GB (51,970 files - complete integration)
│  ├─ Raw: 13 GB (17,592 card images)
│  ├─ Processed: 13 GB (17,592 classification images)
│  ├─ Calibration: 734 MB (1,024 Hailo calibration images)
│  └─ Reference: 106 MB (embeddings + uSearch index)
├─ Profiling: 117 MB
├─ Analytics: ~2 MB
└─ Metadata: ~10 MB
```

### **Model Registry:**
- **Model Package Group:** `pokemon-card-recognition-models`
- **Registered Models:** 2 (Teacher #4, Student Stage 2 #5)

### **Costs:**
- **Training:** $11.50 (one-time, completed)
- **Storage:** $0.135/month (~$1.62/year)
- **After lifecycle optimization:** ~$0.11/month

---

## 🛠️ Common Operations

### **1. View Project Manifest**
```bash
aws s3 cp s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/metadata/project_manifest.json - | jq .
```

### **2. List All Models**
```bash
aws s3 ls s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/models/ --recursive --human-readable
```

### **3. View Analytics Metrics**
```bash
# List metrics files
aws s3 ls s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/analytics/metrics/

# Download cost breakdown
aws s3 cp s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/analytics/metrics/cost_breakdown.csv - | column -t -s,
```

### **4. Check Model Registry**
```bash
# List all registered models
aws sagemaker list-model-packages \
  --model-package-group-name pokemon-card-recognition-models \
  --region us-east-2

# Get specific model details
aws sagemaker describe-model-package \
  --model-package-name arn:aws:sagemaker:us-east-2:943271038849:model-package/pokemon-card-recognition-models/4 \
  --region us-east-2
```

### **5. Download Models Locally**
```bash
# Download teacher model
aws s3 cp s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/models/dinov3-teacher/v1.0/model.tar.gz ./

# Download student PyTorch
aws s3 cp s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/models/efficientnet-student/stage2/v2.0/student_stage2_final.pt ./

# Download Hailo HEF for Raspberry Pi
aws s3 cp s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/models/efficientnet-hailo/v2.1/pokemon_student_efficientnet_lite0_stage2.hef ./
```

### **6. Access Reference Database**
```bash
# Download complete reference database for inference
mkdir -p data/reference
aws s3 sync s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/data/reference/ ./data/reference/

# Or download specific files
aws s3 cp s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/data/reference/embeddings.npy ./
aws s3 cp s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/data/reference/usearch.index ./
aws s3 cp s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/data/reference/metadata.json ./
```

### **7. Access Training Data**
```bash
# List raw card images
aws s3 ls s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/data/raw/card_images/ --recursive | wc -l
# Should show: 17,592 files

# Download raw data (warning: 13 GB)
aws s3 sync s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/data/raw/ ./data/raw/

# Download processed classification dataset
aws s3 sync s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/data/processed/classification/ ./data/processed/

# Download Hailo calibration data (for model compilation)
aws s3 sync s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/models/efficientnet-hailo/calibration/ ./calibration/
```

### **8. Deploy Model to Endpoint**
```python
import boto3
from sagemaker import ModelPackage

role = 'arn:aws:iam::943271038849:role/SageMaker-MarcosAdmin-ExecutionRole'

# Load teacher model from registry
teacher = ModelPackage(
    role=role,
    model_package_arn='arn:aws:sagemaker:us-east-2:943271038849:model-package/pokemon-card-recognition-models/4'
)

# Deploy to endpoint
predictor = teacher.deploy(
    initial_instance_count=1,
    instance_type='ml.c5.xlarge',
    endpoint_name='pokemon-dinov3-teacher'
)
```

---

## 🔒 Security & Access Control

### **Who Has Access:**

1. **Primary Owner:** marcospaulo (AdministratorAccess)
   - Full access to everything via SSO
   - Can manage all resources

2. **Service Account:** SageMaker-MarcosAdmin-ExecutionRole
   - Full admin access to all project resources
   - Used by SageMaker jobs, endpoints, and Studio
   - ARN: `arn:aws:iam::943271038849:role/SageMaker-MarcosAdmin-ExecutionRole`

### **Access Methods:**

1. **AWS Console:** Via SSO as marcospaulo
2. **AWS CLI:** Using your configured credentials
3. **SageMaker Studio:** Using the MarcosAdmin execution role
4. **Programmatic (Python/boto3):** Using the execution role ARN

---

## 📁 Project Structure

```
s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/
│
├── README.md                          # Project overview
│
├── metadata/
│   └── project_manifest.json          # Complete project metadata (v1.1.0)
│
├── models/
│   ├── dinov3-teacher/v1.0/
│   │   └── model.tar.gz              # 5.6 GB - Teacher model
│   │
│   ├── efficientnet-student/
│   │   ├── stage1/v1.0/              # Transitional (not preserved)
│   │   └── stage2/v2.0/
│   │       ├── student_stage2_final.pt    # 74.7 MB - PyTorch
│   │       └── student_stage2_final.onnx  # 22.8 MB - ONNX
│   │
│   └── efficientnet-hailo/v2.1/
│       └── pokemon_student_efficientnet_lite0_stage2.hef  # 13.8 MB - Edge
│
├── data/                             # ✅ COMPLETE DATA INTEGRATION
│   ├── raw/                          # 13 GB - 17,592 original card images
│   │   └── card_images/
│   ├── processed/                    # 13 GB - 17,592 processed for classification training
│   │   └── classification/
│   ├── calibration/                  # 734 MB - 1,024 Hailo calibration images
│   └── reference/                    # 106 MB - Production inference database
│       ├── embeddings.npy            # 51.5 MB - 17,592 x 768 embeddings
│       ├── usearch.index             # 54.0 MB - ARM-optimized vector search
│       ├── index.json                # 652 KB - Row → card_id mapping
│       └── metadata.json             # 543 KB - Card metadata
│
├── experiments/mlflow/               # MLFlow tracking
│   └── experiments_index.json
│
├── profiling/                        # 117 MB - SageMaker Profiler outputs
│   ├── teacher/2026-01-10/          # 44.3 MB
│   └── student_stage2/2026-01-11/   # 72.8 MB
│
├── analytics/
│   ├── dashboards/
│   │   └── dashboard_config.json
│   └── metrics/
│       ├── model_performance.csv
│       ├── compression_metrics.csv
│       ├── cost_breakdown.csv
│       ├── model_lineage.json
│       ├── storage_metrics.csv
│       └── summary.json
│
└── pipelines/
    ├── training/                     # Future: Training automation
    ├── inference/                    # Future: Batch inference
    └── deployment/                   # Future: Deployment pipelines
```

---

## 🎯 Next Steps

### **Immediate:**
1. ✅ Access project via console links above
2. ✅ Verify service account permissions
3. ✅ Review analytics metrics
4. ✅ Check Model Registry entries

### **Development:**
1. 🔄 Set up CloudWatch dashboard using analytics config
2. 🔄 Deploy models to SageMaker endpoints (optional)
3. 🔄 Transfer Hailo HEF to Raspberry Pi 5
4. 🔄 Create SageMaker pipelines for automation

### **Production:**
1. 🚀 Deploy edge inference on Raspberry Pi
2. 🚀 Set up monitoring and alerts
3. 🚀 Create model cards for documentation
4. 🚀 Build real-time card recognition application

---

## 📞 Support & Documentation

- **Complete Cost Breakdown:** `COST_BREAKDOWN.md`
- **Organization Summary:** `ORGANIZATION_COMPLETE.md`
- **Training Guide:** `docs/PRD_06_TRAINING.md`
- **Model Registry Guide:** `docs/MODEL_REGISTRY_GUIDE.md`

---

## 🔍 Verification Script

To verify the service account has full access, run:

```bash
# Test S3 access
aws s3 ls s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/ \
  --profile default  # or specify your profile

# Test SageMaker access
aws sagemaker list-model-packages \
  --model-package-group-name pokemon-card-recognition-models \
  --region us-east-2

# Test IAM access
aws iam get-role \
  --role-name SageMaker-MarcosAdmin-ExecutionRole

# All commands should succeed with full results
```

---

**Your Pokemon Card Recognition project is fully accessible with complete admin permissions!** 🚀

**Primary Service Account:** `SageMaker-MarcosAdmin-ExecutionRole`
**Owner:** Marcos Paulo (marcospaulo)
**Status:** ✅ Full Admin Access Confirmed
