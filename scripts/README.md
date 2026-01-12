# Scripts

Utility scripts for the Pokemon Card Recognition system.

## Embedding Generation

### `generate_embeddings_hailo.py`

Generate embeddings for all Pokemon cards using Hailo 8 NPU.

**What it does:**
1. Downloads all 17,592 card images from S3 (~12.6 GB)
2. Processes each image through Hailo NPU (15.2ms per card)
3. Generates embeddings.npy, usearch.index, index.json, metadata.json
4. Cleans up raw images to free 12.6 GB of space

**Performance:**
- 15.2ms per card on Hailo 8 NPU
- Total time: ~4.5 hours for all 17,592 cards
- Final output: ~111 MB (embeddings + index)

**Run in background:**

```bash
# On Raspberry Pi
cd ~/pokemon-card-recognition
chmod +x scripts/run_embedding_generation.sh
./scripts/run_embedding_generation.sh
```

**Monitor progress:**

```bash
# Watch real-time log
tail -f ~/pokemon-card-recognition/embedding_generation.log

# Check if still running
cat ~/embedding_generation.pid | xargs ps -p

# Check progress (count files downloaded)
ls -1 ~/pokemon-card-recognition/data/raw/card_images/ | wc -l
```

**What gets created:**

```
data/reference/
├── embeddings.npy        # 52 MB - [17592, 768] float32 array
├── usearch.index         # 55 MB - HNSW vector index
├── index.json            # 374 KB - row → card_id mapping
└── metadata.json         # 4.8 MB - card details
```

**Requirements:**
- Hailo 8 NPU installed and working
- AWS CLI configured with S3 access
- ~15 GB free space (temporary, freed after completion)
- Python packages: numpy, opencv-python, usearch, hailo_platform

**Features:**
- ✅ Runs in background with nohup (survives disconnection)
- ✅ Comprehensive logging to file
- ✅ Progress updates every 100 cards
- ✅ Automatic cleanup of raw images
- ✅ Resumes if images already downloaded
- ✅ Error handling and graceful failures

## Notes

- The script automatically cleans up raw images after embedding generation to free space
- If download is interrupted, re-running will resume where it left off (S3 sync is smart)
- Embeddings are saved incrementally (well, all at once after generation, but log shows progress)
- Use existing embeddings from S3 for testing; regenerate only when needed
