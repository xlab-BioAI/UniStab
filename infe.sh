#!/bin/bash
# ============ Edit the following paths for your environment ============
CHECKPOINT=".ckpt"                    # Path to model checkpoint
DATA="/path/to/your/data.csv"         # Input data file
OUTPUT="/path/to/your/output_dir"     # Output directory
NAME="my_experiment"                   # Run name (optional)
BATCH_SIZE=32

python src/inference.py \
    --checkpoint "$CHECKPOINT" \
    --data "$DATA" \
    --batch_size "$BATCH_SIZE" \
    --name "$NAME" \
    --output "$OUTPUT"