#!/bin/bash

# Workflow Automation Script for Online Diffusion Planning
# This script automates the complete development cycle

set -e  # Exit on any error

# Configuration
BERKELEY_USER="ghr"
BERKELEY_HOST="em11.ist.berkeley.edu"
REMOTE_PATH="~/${BERKELEY_USER}/Online-Diffusion-Planning"
COMMIT_MESSAGE="${1:-Auto-commit from workflow automation}"

echo "🚀 Starting Workflow Automation for Online Diffusion Planning"
echo "================================================================"

# Function to run commands and log them
run_cmd() {
    echo "▶️  $1"
    eval "$2"
    if [ $? -eq 0 ]; then
        echo "✅ $1 completed successfully"
    else
        echo "❌ $1 failed"
        exit 1
    fi
}

# Step 1: Check git status
echo ""
echo "📋 Step 1: Checking git status..."
git status

# Step 2: Commit and push to GitHub
echo ""
echo "📤 Step 2: Committing and pushing to GitHub..."
run_cmd "Adding all changes" "git add ."
run_cmd "Committing changes" "git commit -m \"$COMMIT_MESSAGE\""
run_cmd "Pushing to GitHub" "git push origin main"

# Step 3: Update Berkeley server
echo ""
echo "🔄 Step 3: Updating Berkeley server repository..."
run_cmd "Updating Berkeley server" "ssh ${BERKELEY_USER}@${BERKELEY_HOST} \"cd ${REMOTE_PATH} && git pull origin main\""

# Step 4: Run pretrain_planner.py
echo ""
echo "🤖 Step 4: Starting pretrain_planner.py on Berkeley server..."
echo "Training will run in the background. Check status with: ssh ${BERKELEY_USER}@${BERKELEY_HOST} 'ps aux | grep pretrain_planner'"

# Start training in background and capture PID
TRAINING_PID=$(ssh ${BERKELEY_USER}@${BERKELEY_HOST} "cd ${REMOTE_PATH}/Pretrain && nohup python pretrain_planner.py > training_output.log 2>&1 & echo \$!")

echo "Training started with PID: $TRAINING_PID"

# Step 5: Wait for user input to continue
echo ""
echo "⏳ Training is running in the background on Berkeley server."
echo "Press Enter when you want to download the results, or Ctrl+C to exit..."
read -r

# Step 6: Download new and modified files
echo ""
echo "📥 Step 5: Downloading new and modified files from Berkeley server..."

# Download specific files
run_cmd "Downloading model files" "scp ${BERKELEY_USER}@${BERKELEY_HOST}:${REMOTE_PATH}/Pretrain/Kitchen_High_Planner.pt ./Pretrain/"
run_cmd "Downloading stats files" "scp ${BERKELEY_USER}@${BERKELEY_HOST}:${REMOTE_PATH}/Pretrain/Kitchen_High_Planner_stats.pkl ./Pretrain/ 2>/dev/null || echo 'Stats file not found yet'"
run_cmd "Downloading output files" "scp ${BERKELEY_USER}@${BERKELEY_HOST}:${REMOTE_PATH}/Pretrain/output*.txt ./Pretrain/ 2>/dev/null || echo 'Output files not found yet'"

# Step 7: Show training logs
echo ""
echo "📊 Step 6: Fetching recent training logs..."
ssh ${BERKELEY_USER}@${BERKELEY_HOST} "cd ${REMOTE_PATH}/Pretrain && tail -30 training_output.log" 2>/dev/null || echo "Training logs not available yet"

echo ""
echo "🎉 Workflow completed successfully!"
echo ""
echo "📁 Files downloaded to local repository:"
echo "   - Kitchen_High_Planner.pt (trained model)"
echo "   - Kitchen_High_Planner_stats.pkl (training stats)"
echo "   - output*.txt (training logs)"
echo ""
echo "🔍 To check training status: ssh ${BERKELEY_USER}@${BERKELEY_HOST} 'ps aux | grep pretrain_planner'"
echo "📋 To view full logs: ssh ${BERKELEY_USER}@${BERKELEY_HOST} 'cd ${REMOTE_PATH}/Pretrain && tail -f training_output.log'"
