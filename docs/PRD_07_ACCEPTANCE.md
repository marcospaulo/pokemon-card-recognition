# Acceptance Criteria
## PRD_07_ACCEPTANCE.md

**Parent Document:** PRD_01_OVERVIEW.md  
**Purpose:** Consolidated acceptance criteria for all phases

---

## Overview

This document consolidates all acceptance criteria (AC) for the Pokemon Card Recognition System. Each criterion is written in Gherkin format and includes test methodology.

---

## Phase 1: Detection Model

### AC-1.1: Card Detection Accuracy
```gherkin
GIVEN a frame containing a visible Pokemon card
WHEN the detection model processes the frame
THEN a detection MUST be returned with confidence ≥ 0.7
AND the bounding box IoU with ground truth MUST be ≥ 0.85
```

**Test Method:**
- Run inference on 200 test images with ground truth bboxes
- Calculate IoU for each detection
- Pass if ≥95% of detections meet criteria

### AC-1.2: Keypoint Accuracy
```gherkin
GIVEN a detected Pokemon card
WHEN extracting corner keypoints
THEN each keypoint MUST be within 5 pixels of the actual corner
AND each keypoint confidence MUST be ≥ 0.5
```

**Test Method:**
- Manually annotate 100 test images with exact corner coordinates
- Measure pixel distance for each detected keypoint
- Pass if ≥90% of keypoints are within tolerance

### AC-1.3: False Positive Rejection (Empty Frame)
```gherkin
GIVEN a frame with NO Pokemon card (empty table, hand, other objects)
WHEN the detection model processes the frame
THEN NO detection MUST be returned with confidence ≥ 0.5
```

**Test Method:**
- Test with 100 negative images (tables, hands, other objects)
- Count false positives with confidence ≥ 0.5
- Pass if false positive rate < 1%

### AC-1.4: Rotation Handling
```gherkin
GIVEN a Pokemon card rotated between 0° and 45°
WHEN the detection model processes the frame
THEN detection confidence MUST remain ≥ 0.7
AND keypoints MUST correctly identify corners despite rotation
```

**Test Method:**
- Test with cards at 0°, 15°, 30°, 45° rotation
- Verify detection and keypoint accuracy at each angle

### AC-1.5: Partial Visibility
```gherkin
GIVEN a Pokemon card with up to 20% outside the frame edge
WHEN the detection model processes the frame
THEN detection SHOULD succeed with confidence ≥ 0.6
AND visible corners MUST have keypoint confidence ≥ 0.5
```

**Test Method:**
- Test with cards partially cropped (10%, 15%, 20%)
- Measure detection rate and keypoint accuracy

### AC-1.6: Detection Speed
```gherkin
GIVEN the model deployed on IMX500
WHEN processing continuous video at 30 FPS
THEN inference time MUST be < 10ms per frame
AND frame rate MUST NOT drop below 25 FPS
```

**Test Method:**
- Run 1000 frames continuously
- Measure average inference time and frame rate
- Pass if avg inference < 10ms and sustained FPS ≥ 25

---

## Phase 2: Embedding Model

### AC-2.1: Self-Retrieval
```gherkin
GIVEN an embedding computed from a card image
WHEN searching the reference database containing that card
THEN the SAME card MUST be the nearest neighbor
AND distance MUST be < 0.1
```

**Test Method:**
- For each card in database, compute embedding and search
- Verify self is top result with distance < 0.1
- Pass if 100% self-retrieval

### AC-2.2: Clean Image Accuracy
```gherkin
GIVEN a clean, well-lit card image (not from training set)
WHEN computing embedding and searching database
THEN the correct card MUST be in top-1 result
AND distance MUST be < 0.2
```

**Test Method:**
- Test with held-out test set (different photos of same cards)
- Measure top-1 accuracy
- Pass if ≥ 95% top-1 accuracy

### AC-2.3: Occluded Image Handling
```gherkin
GIVEN a card image with 20% occlusion (fingers, shadow, etc.)
WHEN computing embedding and searching database
THEN the correct card MUST be in top-3 results
AND distance to correct card MUST be < 0.4
```

**Test Method:**
- Synthetically occlude test images (20% coverage)
- Measure top-3 accuracy
- Pass if ≥ 85% top-3 accuracy

### AC-2.4: Glare Handling
```gherkin
GIVEN a holo or gold card with visible specular glare
WHEN computing embedding and searching database
THEN the correct card MUST be in top-5 results
```

**Test Method:**
- Test with holo/gold cards photographed with visible glare
- Measure top-5 accuracy
- Pass if ≥ 80% top-5 accuracy

### AC-2.5: Unknown Input Rejection
```gherkin
GIVEN an image that is NOT a Pokemon card
WHEN computing embedding and searching database
THEN distance to nearest card MUST be > 0.5
```

**Test Method:**
- Test with 100 non-card images (tables, random objects)
- Verify all nearest neighbor distances > 0.5
- Pass if 100% of non-cards are rejected

### AC-2.6: Embedding Inference Speed
```gherkin
GIVEN the model deployed on Hailo 8
WHEN processing a single 224×224 image
THEN inference time MUST be < 25ms
```

**Test Method:**
- Run 1000 inferences, measure latency
- Pass if average latency < 25ms and p99 < 35ms

### AC-2.7: Embedding Consistency
```gherkin
GIVEN two images of the SAME card under different conditions
  (lighting, angle, minor occlusion)
WHEN computing embeddings for both
THEN cosine similarity MUST be > 0.8
```

**Test Method:**
- Collect 3+ images per card under varying conditions
- Compute pairwise similarities within same card
- Pass if ≥ 90% of pairs have similarity > 0.8

### AC-2.8: Embedding Discrimination
```gherkin
GIVEN embeddings of two DIFFERENT cards
WHEN computing cosine similarity
THEN similarity MUST be < 0.5
```

**Test Method:**
- Sample 1000 random card pairs (different cards)
- Compute similarities
- Pass if ≥ 99% of pairs have similarity < 0.5

---

## Phase 3: Reference Database

### AC-3.1: Database Loading
```gherkin
GIVEN a properly formatted reference database
WHEN loading the database into memory
THEN loading MUST complete successfully
AND embedding count MUST match index count
AND loading time MUST be < 5 seconds on Raspberry Pi 5
```

**Test Method:**
- Load full 17k card database
- Verify counts match, measure load time
- Pass if load time < 5s

### AC-3.2: Search Speed
```gherkin
GIVEN a query embedding
WHEN searching 17,592 card database
THEN search time MUST be < 10ms on Raspberry Pi 5
```

**Test Method:**
- Run 1000 random searches
- Measure average search time
- Pass if average < 10ms

### AC-3.3: Search Accuracy
```gherkin
GIVEN a query embedding for a known card
WHEN searching the database
THEN the correct card MUST be returned as top result
AND distance MUST be < 0.1 for exact embedding match
```

**Test Method:**
- Use stored embeddings as queries
- Verify self-search returns correct card
- Pass if 100% accuracy

### AC-3.4: Threshold Filtering
```gherkin
GIVEN a query with distance_threshold=0.4
WHEN no cards are within threshold
THEN empty results MUST be returned
AND system MUST NOT crash or return invalid results
```

**Test Method:**
- Search with random/garbage embeddings
- Verify proper empty result handling

### AC-3.5: Memory Usage
```gherkin
GIVEN the database loaded on Raspberry Pi 5
THEN total memory usage MUST be < 200MB
AND system MUST remain responsive for other tasks
```

**Test Method:**
- Monitor memory usage during operation
- Run concurrent tasks (camera, inference)
- Pass if total memory < 200MB

### AC-3.6: Database Integrity
```gherkin
GIVEN the database files (embeddings, metadata, index)
WHEN validating integrity
THEN embedding row count MUST equal index count
AND all card_ids in index MUST exist in metadata
AND no duplicate card_ids MUST exist
```

**Test Method:**
- Run integrity validation script
- Pass if all checks pass

---

## Phase 4: Matching Pipeline

### AC-4.1: No Card Rejection
```gherkin
GIVEN no Pokemon card in the camera frame
WHEN processing the frame through the pipeline
THEN status MUST be "no_card"
AND card_match MUST be None
AND system MUST NOT predict any card
```

**Test Method:**
- Point camera at empty table for 10 seconds
- Verify all frames return "no_card"
- Pass if 100% correct rejection

### AC-4.2: Unknown Card Detection
```gherkin
GIVEN a card that is NOT in the reference database
WHEN processing frames through the pipeline
THEN status MUST be "unknown_card" after stabilization
AND embedding_distance MUST be > 0.4
```

**Test Method:**
- Use cards from sets not in database
- Verify system reports "unknown_card"

### AC-4.3: Known Card Confirmation
```gherkin
GIVEN a card that IS in the reference database
AND the card is clearly visible without major occlusion
WHEN processing 3+ consecutive frames
THEN status MUST transition to "confirmed"
AND card_match.name MUST match the actual card
AND embedding_distance MUST be < 0.25
```

**Test Method:**
- Test with 100 known cards
- Verify correct identification within 5 frames
- Pass if ≥ 95% correctly identified

### AC-4.4: Temporal Stability
```gherkin
GIVEN predictions are unstable (different cards each frame)
WHEN the system is stabilizing
THEN status MUST remain "stabilizing"
AND NO confirmed result MUST be emitted
```

**Test Method:**
- Rapidly move card to create unstable predictions
- Verify no premature confirmations

### AC-4.5: End-to-End Latency
```gherkin
GIVEN the complete pipeline running on target hardware
WHEN processing frames continuously
THEN total pipeline latency MUST be < 100ms
AND effective frame rate MUST be ≥ 10 FPS
```

**Test Method:**
- Measure end-to-end latency for 1000 frames
- Pass if average < 100ms, min FPS ≥ 10

### AC-4.6: Glare Handling (Pipeline)
```gherkin
GIVEN a holo or gold card with visible glare
WHEN processing frames through the pipeline
THEN system SHOULD reach "confirmed" status
AND correct card SHOULD be in top_matches
AND glare_detected flag SHOULD be True
```

**Test Method:**
- Test with 50 holo/gold cards under normal lighting
- Measure confirmation rate
- Pass if ≥ 80% correctly identified

### AC-4.7: Context Contamination (Pipeline)
```gherkin
GIVEN a card held by fingers (up to 20% occluded)
WHEN processing frames through the pipeline
THEN system SHOULD reach "confirmed" status
AND correct card SHOULD be identified
```

**Test Method:**
- Test with cards held in various grips
- Measure identification accuracy
- Pass if ≥ 85% correctly identified

---

## System-Level Acceptance Criteria

### AC-S.1: Cold Start Time
```gherkin
GIVEN the system starting from power-off
WHEN the application starts
THEN the system MUST be ready for recognition
  within 30 seconds of application launch
```

**Test Method:**
- Time from `python main.py` to first successful frame
- Pass if < 30 seconds

### AC-S.2: Continuous Operation
```gherkin
GIVEN the system running continuously
WHEN processing frames for 1 hour
THEN there MUST be no memory leaks (< 10% memory growth)
AND there MUST be no crashes or hangs
AND accuracy MUST remain consistent
```

**Test Method:**
- Run overnight stress test (8+ hours)
- Monitor memory usage, crash count, accuracy

### AC-S.3: Error Recovery
```gherkin
GIVEN a temporary hardware error (camera disconnect, Hailo timeout)
WHEN the error is resolved
THEN the system MUST recover automatically
AND MUST NOT require restart
```

**Test Method:**
- Simulate hardware errors
- Verify automatic recovery

### AC-S.4: Logging and Debugging
```gherkin
GIVEN the system in operation
WHEN an error or anomaly occurs
THEN relevant information MUST be logged
AND logs MUST be sufficient for debugging
```

**Test Method:**
- Review logs after intentional error injection
- Verify error details are captured

---

## Test Summary Table

| Phase | Criterion | Priority | Automated | Status |
|-------|-----------|----------|-----------|--------|
| 1 | AC-1.1 Detection Accuracy | P0 | Yes | ⬜ |
| 1 | AC-1.2 Keypoint Accuracy | P0 | Yes | ⬜ |
| 1 | AC-1.3 False Positive Rejection | P0 | Yes | ⬜ |
| 1 | AC-1.4 Rotation Handling | P1 | Yes | ⬜ |
| 1 | AC-1.5 Partial Visibility | P1 | Yes | ⬜ |
| 1 | AC-1.6 Detection Speed | P0 | Yes | ⬜ |
| 2 | AC-2.1 Self-Retrieval | P0 | Yes | ⬜ |
| 2 | AC-2.2 Clean Image Accuracy | P0 | Yes | ⬜ |
| 2 | AC-2.3 Occluded Image Handling | P0 | Yes | ⬜ |
| 2 | AC-2.4 Glare Handling | P1 | Yes | ⬜ |
| 2 | AC-2.5 Unknown Input Rejection | P0 | Yes | ⬜ |
| 2 | AC-2.6 Embedding Inference Speed | P0 | Yes | ⬜ |
| 2 | AC-2.7 Embedding Consistency | P1 | Yes | ⬜ |
| 2 | AC-2.8 Embedding Discrimination | P1 | Yes | ⬜ |
| 3 | AC-3.1 Database Loading | P0 | Yes | ⬜ |
| 3 | AC-3.2 Search Speed | P0 | Yes | ⬜ |
| 3 | AC-3.3 Search Accuracy | P0 | Yes | ⬜ |
| 3 | AC-3.4 Threshold Filtering | P1 | Yes | ⬜ |
| 3 | AC-3.5 Memory Usage | P0 | Yes | ⬜ |
| 3 | AC-3.6 Database Integrity | P0 | Yes | ⬜ |
| 4 | AC-4.1 No Card Rejection | P0 | Yes | ⬜ |
| 4 | AC-4.2 Unknown Card Detection | P0 | Yes | ⬜ |
| 4 | AC-4.3 Known Card Confirmation | P0 | Yes | ⬜ |
| 4 | AC-4.4 Temporal Stability | P1 | Yes | ⬜ |
| 4 | AC-4.5 End-to-End Latency | P0 | Yes | ⬜ |
| 4 | AC-4.6 Glare Handling (Pipeline) | P1 | Manual | ⬜ |
| 4 | AC-4.7 Context Contamination | P1 | Manual | ⬜ |
| S | AC-S.1 Cold Start Time | P1 | Yes | ⬜ |
| S | AC-S.2 Continuous Operation | P1 | Yes | ⬜ |
| S | AC-S.3 Error Recovery | P2 | Manual | ⬜ |
| S | AC-S.4 Logging and Debugging | P2 | Manual | ⬜ |

**Priority Key:**
- P0: Must pass for phase completion
- P1: Should pass, minor issues acceptable
- P2: Nice to have

---

## Definition of Done

### Per Phase
A phase is considered complete when:
1. All P0 acceptance criteria pass
2. ≥80% of P1 acceptance criteria pass
3. Code is reviewed and merged
4. Documentation is updated
5. Models are exported and deployable

### Full System
The system is considered production-ready when:
1. All phases complete per above criteria
2. End-to-end integration tests pass
3. 8-hour stress test completes without issues
4. User acceptance testing complete
5. Performance meets all targets

---

## Sign-Off

| Phase | Reviewer | Date | Signature |
|-------|----------|------|-----------|
| 1 - Detection | | | |
| 2 - Embedding | | | |
| 3 - Database | | | |
| 4 - Pipeline | | | |
| System Integration | | | |
