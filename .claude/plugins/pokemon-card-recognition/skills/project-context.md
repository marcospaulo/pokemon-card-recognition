---
skill_name: pokemon-card-recognition:context
description: Quick reference for Pokemon Card Recognition project - links to detailed sub-skills
tags: [sagemaker, ml, pokemon, overview]
---

# Pokemon Card Recognition - Quick Reference

## Project Overview

**What:** ML models to recognize Pokemon trading cards using image embeddings and vector search

**Architecture:** Teacher (DINOv3) → Student (EfficientNet) → Edge (Hailo NPU)

**Dataset:** 17,592 images, 2,199 unique cards

**Status:** ✅ Training complete, all models organized and documented

---

## Quick Access

### AWS Resources
- **Account:** 943271038849 (us-east-2)
- **S3 Bucket:** `s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/`
- **Model Registry:** `pokemon-card-recognition-models` (2 approved models)
- **Service Account:** `SageMaker-MarcosAdmin-ExecutionRole` (Full Admin)
- **User Profile:** marcospaulo (Domain: d-slzqikvnlai2)

### Console Links
- **SageMaker Studio:** https://d-slzqikvnlai2.studio.us-east-2.sagemaker.aws
- **S3 Project:** https://s3.console.aws.amazon.com/s3/buckets/pokemon-card-training-us-east-2?prefix=project/pokemon-card-recognition/
- **Model Registry:** https://console.aws.amazon.com/sagemaker/home?region=us-east-2#/model-package-groups/pokemon-card-recognition-models

---

## Essential Commands

### View Project Manifest
```bash
aws s3 cp s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/metadata/project_manifest.json - | jq .
```

### Download Production Model (for edge)
```bash
aws s3 cp s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/models/efficientnet-hailo/v2.1/pokemon_student_efficientnet_lite0_stage2.hef ./
```

### Download Reference Database
```bash
aws s3 sync s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/data/reference/ ./data/reference/
```

### Verify Access (10 tests)
```bash
bash scripts/verify_project_access.sh
```

---

## Project Structure Summary

```
S3: s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/
├── models/              # 5.7 GB - 4 model variants
├── data/                # 26.7 GB - Images + embeddings + index
├── profiling/           # 117 MB - Training metrics
├── analytics/           # 2 MB - Performance metrics
└── metadata/            # Project manifest v1.1.0
```

**Total:** ~31.7 GB (53,068 objects)

---

## Cost Summary

- **Training:** $11.50 USD (one-time, completed)
- **Storage:** $0.135/month (~$1.62/year)
- **Edge vs Cloud:** Raspberry Pi saves $777/year

---

## Detailed Sub-Skills

Load these skills when you need specific details:

### 📦 AWS Resources & Access
**Skill:** `aws-resources.md`
**When to use:** Need S3 structure, IAM permissions, bucket policies, console links, service account details

### 🤖 Model Details & Training
**Skill:** `model-details.md`
**When to use:** Need model architectures, parameters, compression ratios, training configs, performance metrics

### ⚙️ Operations & Workflows
**Skill:** `operations.md`
**When to use:** Need commands for downloading models, deploying endpoints, training new models, accessing data

### 💰 Cost & Optimization
**Skill:** `cost-optimization.md`
**When to use:** Need detailed cost breakdown, lifecycle policies, storage optimization, cloud vs edge comparison

---

## Key Documentation

**In Repository:**
- `ORGANIZATION_COMPLETE.md` - Complete project summary
- `COST_BREAKDOWN.md` - Detailed cost analysis
- `PROJECT_ACCESS.md` - All access links and commands
- `MARCOSPAULO_ACCESS_SUMMARY.md` - Service account permissions

**In S3:**
- `metadata/project_manifest.json` - Complete metadata (v1.1.0)
- `analytics/metrics/` - Model performance, compression, cost CSVs

---

## Important Notes

- **SageMaker Profile Selection:** When prompted, choose "marcospaulo" - this gives you full admin access
- **Student Stage 1:** Marked as transitional, checkpoints not preserved (only Stage 2 saved)
- **Lifecycle Policies:** Training outputs → Glacier after 90 days, profiling → Glacier after 180 days
- **Edge Deployment:** Raspberry Pi 5 + Hailo-8L (30ms inference, $777/year savings vs cloud)

---

**Last Updated:** 2026-01-12
**Status:** ✅ Complete
**Next Steps:** Deploy to Raspberry Pi or create SageMaker endpoint
