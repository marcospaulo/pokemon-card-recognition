#!/bin/bash
# Run embedding generation in background with nohup
# This script will survive disconnection and continue running

cd ~/pokemon-card-recognition

# Activate virtual environment
source venv/bin/activate

# Run Python script in background
nohup python3 scripts/generate_embeddings_hailo.py > ~/embedding_generation_stdout.log 2>&1 &

# Save PID
PID=$!
echo $PID > ~/embedding_generation.pid

echo "✅ Embedding generation started in background"
echo "   PID: $PID"
echo "   Log: ~/pokemon-card-recognition/embedding_generation.log"
echo "   Stdout: ~/embedding_generation_stdout.log"
echo ""
echo "Monitor progress:"
echo "   tail -f ~/pokemon-card-recognition/embedding_generation.log"
echo ""
echo "Check if running:"
echo "   ps -p $PID"
echo ""
echo "Kill process:"
echo "   kill $PID"
