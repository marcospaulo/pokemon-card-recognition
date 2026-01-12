# Pokemon Card Recognition - Claude Code Plugin

This plugin provides comprehensive context for the Pokemon Card Recognition ML project, making it easy for any Claude Code agent to quickly understand the project structure, AWS resources, and common operations.

## Plugin Structure

```
.claude/plugins/pokemon-card-recognition/
├── plugin.json                     # Plugin manifest
├── README.md                       # This file
└── skills/
    ├── project-context.md          # Main skill - Quick reference (lightweight)
    ├── aws-resources.md            # Detailed AWS resources and IAM permissions
    ├── model-details.md            # Model architectures and training configs
    ├── operations.md               # Commands and workflows
    └── cost-optimization.md        # Cost breakdown and optimization strategies
```

## How to Use

### Main Skill: Quick Reference

The **project-context** skill is your starting point. It provides:
- Project overview and status
- Quick access links (S3, SageMaker, Model Registry)
- Essential commands (download models, verify access)
- Project structure summary
- Cost summary
- References to detailed sub-skills

**When to use:** Always start here to get oriented.

### Sub-Skills: Detailed Information

Load these skills when you need specific details:

#### 📦 aws-resources.md
**When to use:**
- Need complete S3 directory structure
- Want to understand IAM permissions
- Need to verify service account access
- Looking for console links
- Understanding lifecycle policies

**Contains:**
- Complete S3 bucket layout with sizes
- SageMaker resources (Model Registry, user profiles)
- IAM role permissions (10 managed policies + 2 custom)
- S3 bucket policies and tags
- Lifecycle policy configurations
- Console access links
- Access verification commands

#### 🤖 model-details.md
**When to use:**
- Need model architecture details
- Want to understand training configurations
- Need compression metrics
- Understanding model lineage (teacher → student)
- Looking for model performance specs

**Contains:**
- DINOv3 Teacher architecture and training config
- EfficientNet Student Stage 1 & 2 details
- Hailo edge model specifications
- Compression ratios (64.7x parameter reduction)
- Training costs by component
- Reference database generation process
- Model performance comparison

#### ⚙️ operations.md
**When to use:**
- Need commands to download models
- Want to deploy to SageMaker endpoint
- Need to access training data
- Want to transfer models to Raspberry Pi
- Need verification or testing commands

**Contains:**
- Download commands (models, data, reference database)
- SageMaker operations (Model Registry, training jobs, endpoints)
- Data operations (raw, processed, calibration)
- Profiling data access
- CloudWatch logs commands
- Raspberry Pi deployment workflow
- Training new models commands
- Verification and testing commands

#### 💰 cost-optimization.md
**When to use:**
- Need detailed cost breakdown
- Want to understand lifecycle policies
- Comparing cloud vs edge deployment
- Looking for optimization strategies
- Setting up cost monitoring

**Contains:**
- Complete cost breakdown (training: $11.50, storage: $0.30/month)
- Lifecycle policy explanations (what they are, how they work)
- Storage class comparison (S3 Standard vs Glacier)
- Cloud vs Edge deployment cost analysis ($1,762/year cloud vs $23/year edge)
- Optimization strategies (spot instances, compression, etc.)
- Cost monitoring setup (CloudWatch alarms)

## Usage Examples

### Example 1: Getting Oriented
```
You: "What's the status of the Pokemon Card Recognition project?"

Claude: *Loads project-context.md*
"The project is complete! We have 4 model variants trained:
- Teacher: DINOv3 (304M params)
- Student Stage 2: EfficientNet (4.7M params, 64.7x compressed)
- Hailo Edge: INT8 quantized (13.8 MB)

Training cost: $11.50 (one-time)
Storage cost: $0.30/month

All models are in S3 and registered in Model Registry."
```

### Example 2: Downloading Models
```
You: "How do I download the Hailo model for Raspberry Pi?"

Claude: *Loads operations.md*
"Here's the command to download the Hailo HEF:

aws s3 cp s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/models/efficientnet-hailo/v2.1/pokemon_student_efficientnet_lite0_stage2.hef ./

Then transfer to Pi:
scp pokemon_student_efficientnet_lite0_stage2.hef pi@raspberrypi:/home/pi/models/"
```

### Example 3: Understanding Costs
```
You: "Why is the training cost $11.50? Break it down for me."

Claude: *Loads cost-optimization.md*
"The $11.50 is one-time training compute cost, not storage:

- Teacher: $4.00 (8xA100, 12 min)
- Student Stage 1: $4.00 (8xA100, 15 min)
- Student Stage 2: $3.00 (8xA100, 10 min)
- Hailo compilation: $0.50 (m5.2xlarge, 60 min)

Storage is separate: $0.30/month ongoing."
```

### Example 4: Verifying Access
```
You: "How do I verify I have access to everything?"

Claude: *Loads aws-resources.md*
"Run this script to verify all 10 access tests:

bash scripts/verify_project_access.sh

This checks:
- S3 read/write access
- SageMaker Model Registry access
- IAM role permissions
- CloudWatch logs
- And 6 more tests

Expected: All 10 tests pass ✓"
```

## Benefits of This Plugin

### ✅ Efficient Context Usage
- Main skill is lightweight (~150 lines)
- Only load detailed skills when needed
- Saves tokens, reduces context usage

### ✅ Easy to Maintain
- Each skill is self-contained
- Update one skill without affecting others
- Clear separation of concerns

### ✅ Fast Onboarding
- Future agents instantly understand project
- No need to re-explain structure
- Complete documentation in one place

### ✅ Comprehensive Coverage
- AWS resources and permissions
- Model architectures and training
- Common operations and commands
- Cost breakdown and optimization

## Future Enhancements

Potential additions:
- **Commands:** Add slash commands for common operations (e.g., `/download-model`, `/verify-access`)
- **Agents:** Create specialized agents (e.g., `deployment-agent`, `cost-analyzer`)
- **Hooks:** Add hooks for automated cost monitoring or lifecycle policy updates

## Version History

**v1.0.0** (2026-01-12)
- Initial plugin creation
- 5 skills: project-context, aws-resources, model-details, operations, cost-optimization
- Complete project documentation
- Lightweight main skill with detailed sub-skills

---

**Last Updated:** 2026-01-12
**Plugin Version:** 1.0.0
**Project Status:** ✅ Complete
**Total Project Cost:** $11.50 (training) + $0.30/month (storage)
