# Pokemon Card Recognition System
## Product Requirements Document (PRD)

**Version:** 1.0  
**Created:** January 2025  
**Author:** Marcos / Grail Seeker  
**Status:** Draft for Implementation

---

## Executive Summary

Build a robust, local-first Pokemon card recognition system using a **two-stage AI architecture** on Raspberry Pi 5 with Sony IMX500 AI Camera and Hailo-8L accelerator. Stage 1 (IMX500) detects cards with low power consumption; Stage 2 (Hailo-8L) performs high-quality embedding inference. The system uses **embedding-based similarity search** instead of classification, enabling natural rejection of non-card inputs and graceful handling of unknown cards.

---

## Problem Statement

### Current Issues

1. **No "None" Class:** Classification models must pick one of 17,592 cards, even for empty tables (produces 50-60% confidence garbage predictions)

2. **Context Contamination:** Extra background/fingers cause 10% accuracy drop

3. **Glare Instability:** Holo/gold cards cause oscillating 60-70% predictions

4. **Brittleness:** Model fails when card isn't perfectly cropped or lit

### Root Cause

**Classification is the wrong paradigm.** Forcing 17,592 mutually exclusive classes means:
- No natural way to say "this isn't a card"
- No graceful handling of new/unknown cards
- Confidence scores are misleading (softmax must sum to 1.0)

---

## Proposed Solution: Embedding-Based Recognition

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TWO-STAGE AI ARCHITECTURE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STAGE 1: DETECTION (Sony IMX500 Camera - Always On, <1W)                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  YOLOv8n: "Is there a card? Where is it?"                           │   │
│  │  - Distilled from pre-trained YOLO11n-OBB (10k Pokemon cards)       │   │
│  │  - Output: Bounding box (x, y, w, h) + confidence                   │   │
│  │  - Runs at: 30+ FPS on IMX500 on-sensor processing                  │   │
│  │  - Power: <1W continuous                                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  PREPROCESSING (Raspberry Pi 5 CPU)                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Crop card region → Resize to 224×224 → Normalize                   │   │
│  │  Output: Clean card image ready for embedding                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  STAGE 2: EMBEDDING (Hailo-8L Accelerator - On Demand, 2-4W)               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  ConvNeXt-Tiny: Image → 768-dim embedding vector                    │   │
│  │  - Distilled from DINOv3-ViT-L/16 teacher model                     │   │
│  │  - NOT classification - pure feature extraction                     │   │
│  │  - Output: Dense vector representing card visual features           │   │
│  │  - Runs at: <25ms on Hailo-8L INT8                                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  MATCHING (Raspberry Pi 5 CPU/RAM)                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Compare embedding against reference database (17k card embeddings) │   │
│  │  - Cosine similarity via uSearch HNSW index                         │   │
│  │  - Find top-K nearest neighbors (<1ms)                              │   │
│  │  - Output: Matched card ID + similarity score + confidence          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  VALIDATION (Raspberry Pi 5 CPU)                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Temporal smoothing + Distance threshold checking                   │   │
│  │  Output: Final result OR "Unknown card" OR "Not a card"             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why Embeddings Solve Your Problems

| Problem | Classification | Embedding |
|---------|---------------|-----------|
| Empty table | Forces prediction (Zekrom 57%) | No close match → "Not a card" |
| New/unknown card | Misclassifies to nearest trained class | Low similarity → "Unknown card" |
| Confidence meaning | Meaningless (softmax artifact) | Actual distance metric |
| Adding new cards | Retrain entire model | Just add embedding to database |
| Partial occlusion | Feature corruption | Partial features still match |

### Key Insight

**The embedding model doesn't decide what card it is.** It only converts image → vector. The *database lookup* decides the match. This separation means:

- Model is reusable across new card sets
- No retraining needed when new cards release
- Just add new embeddings to reference database
- Natural "I don't know" when nothing matches

---

## System Components

### Hardware

| Component | Role | Model/Spec |
|-----------|------|------------|
| Camera | Capture + Detection | Raspberry Pi AI Camera (IMX500) |
| Compute | Orchestration + Preprocessing | Raspberry Pi 5 (8GB) |
| AI Accelerator | Embedding inference | Hailo 8 (26 TOPS) |
| Storage | Reference database | Local JSON/SQLite (future: pgvector) |

### Software

| Component | Technology | Purpose |
|-----------|------------|---------|
| Detection Model | YOLOv8n-pose | Card localization + corner detection |
| Embedding Model | ViT-B/16 or CLIP | Visual feature extraction |
| Reference DB | JSON → SQLite → pgvector | Card embedding storage |
| Inference Runtime | Hailo Runtime + picamera2 | Hardware integration |
| Training | Amazon SageMaker | Model training pipeline |

---

## Project Phases

### Phase 1: Detection Model
**Goal:** Reliably detect cards and extract corner keypoints  
**Duration:** 1-2 weeks  
**Deliverable:** YOLOv8n-pose model deployed on IMX500

### Phase 2: Embedding Model  
**Goal:** Convert card images to discriminative embeddings  
**Duration:** 2-3 weeks  
**Deliverable:** ViT encoder deployed on Hailo 8

### Phase 3: Reference Database
**Goal:** Pre-compute embeddings for all 17k cards  
**Duration:** 1 week  
**Deliverable:** JSON file with card_id → embedding mapping

### Phase 4: Matching Pipeline
**Goal:** Fast similarity search with rejection logic  
**Duration:** 1 week  
**Deliverable:** Inference pipeline with temporal smoothing

### Phase 5: Integration & Testing
**Goal:** End-to-end system on Raspberry Pi  
**Duration:** 1-2 weeks  
**Deliverable:** Working local system

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Empty table rejection | 100% | Should NEVER output a card prediction |
| Clean card accuracy (Top-1) | ≥95% | Correct card in first result |
| Clean card accuracy (Top-3) | ≥99% | Correct card in top 3 results |
| Partial occlusion accuracy | ≥85% | 20% of card obscured |
| Glare handling accuracy | ≥80% | Holo/gold cards with visible glare |
| End-to-end latency | <100ms | Frame capture to result |
| Unknown card detection | ≥90% | Cards not in database flagged correctly |

---

## Non-Goals (Out of Scope)

- Cloud connectivity (local-first)
- Mobile app (Raspberry Pi only for now)
- Price lookup (future enhancement)
- Inventory management integration (future)
- Multiple simultaneous cards (single card for v1)

---

## Dependencies

### External Services
- Amazon SageMaker (training only)
- Hailo Model Zoo (model conversion)

### Data Requirements
- 17,592 card reference images (you have this)
- 500-1000 detection training images (to be labeled)
- Card metadata JSON (name, set, number)

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Hailo ViT compatibility | Medium | High | Test early, have ONNX fallback |
| Embedding discrimination | Low | High | Use contrastive loss, not classification |
| IMX500 model conversion | Medium | Medium | Follow Sony's documentation closely |
| Reference DB size | Low | Low | 17k × 768 floats = ~50MB (fits in RAM) |

---

## Document Index

| Document | Purpose |
|----------|---------|
| `PRD_01_OVERVIEW.md` | This document - system overview |
| `PRD_02_DETECTION.md` | Phase 1: Detection model specs |
| `PRD_03_EMBEDDING.md` | Phase 2: Embedding model specs |
| `PRD_04_DATABASE.md` | Phase 3: Reference database design |
| `PRD_05_PIPELINE.md` | Phase 4: Matching pipeline |
| `PRD_06_TRAINING.md` | SageMaker training guide |
| `PRD_07_ACCEPTANCE.md` | Acceptance criteria for all phases |

---

## Approval

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Product Owner | Marcos | | |
| Technical Lead | | | |

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | Jan 2025 | Claude | Initial draft |
