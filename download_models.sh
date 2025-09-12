#!/bin/bash

# Create directory if it doesn't exist
mkdir -p /Users/leo/Desktop/Checkpoints

# Download files from 37000 to 36000 with step 100
for step in {37000..36000..-100}; do
    echo "Downloading step $step..."
    scp ghr@em11.ist.berkeley.edu:~/Online-Diffusion-Planning/Pretrain/Pretrain/Checkpoints/Kitchen_Medium_Planner_${step}.pt /Users/leo/Desktop/Checkpoints/Kitchen_Medium_Planner_${step}.pt
    
    # Check if download was successful
    if [ $? -eq 0 ]; then
        echo "✅ Successfully downloaded step $step"
    else
        echo "❌ Failed to download step $step"
    fi
    echo "---"
done

echo "Download complete!"
