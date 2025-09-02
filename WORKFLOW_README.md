# 🚀 Workflow Automation for Online Diffusion Planning

This repository now includes automated workflow scripts that streamline your entire development cycle: **modify code locally → upload to Berkeley server → run training → download results**.

## 📋 What the Workflow Does

1. **🔍 Check Git Status** - Shows what changes will be committed
2. **📤 Commit & Push** - Automatically commits and pushes to GitHub
3. **🔄 Update Berkeley Server** - Pulls latest code to your remote terminal
4. **🤖 Run Training** - Starts `pretrain_planner.py` on Berkeley server
5. **📥 Download Results** - Brings new model files and logs back to local
6. **📊 Show Logs** - Displays recent training output

## 🛠️ Available Scripts

### **Option 1: Python Script (Recommended)**
```bash
python workflow_automation.py [options]
```

**Features:**
- Full automation with progress monitoring
- Configurable options
- Better error handling
- Training completion detection

**Options:**
```bash
python workflow_automation.py --help

# Custom commit message
python workflow_automation.py -m "Fixed diffusion sampling bug"

# Run training in background (don't wait)
python workflow_automation.py --no-wait

# Custom Berkeley server details
python workflow_automation.py --berkeley-user your_username --berkeley-host your_host
```

### **Option 2: Shell Script (Simple)**
```bash
./workflow.sh [commit_message]
```

**Features:**
- Simple one-liner execution
- Interactive (waits for user input to download results)
- Easy to modify and customize

**Usage:**
```bash
# Basic usage
./workflow.sh

# With custom commit message
./workflow.sh "Added new training parameters"
```

## 🎯 Quick Start

### **First Time Setup**
1. **Ensure SSH access** to Berkeley server:
   ```bash
   ssh ghr@em11.ist.berkeley.edu
   ```

2. **Make script executable** (if not already done):
   ```bash
   chmod +x workflow.sh
   ```

### **Daily Workflow**
1. **Modify your code** locally (e.g., in `Pretrain/pretrain_planner.py`)

2. **Run the automation**:
   ```bash
   # Option 1: Python script (recommended)
   python workflow_automation.py -m "Updated learning rate and batch size"
   
   # Option 2: Shell script
   ./workflow.sh "Updated learning rate and batch size"
   ```

3. **Wait for completion** or let it run in background

4. **Results automatically downloaded** to your local repository

## 📁 Files Downloaded

After training completes, these files are automatically downloaded:

- **`Pretrain/Kitchen_High_Planner.pt`** - Trained diffusion planner model
- **`Pretrain/Kitchen_High_Planner_stats.pkl`** - Training statistics and metadata
- **`Pretrain/output*.txt`** - Training logs and output files

## 🔍 Monitoring Training

### **Check Training Status**
```bash
ssh ghr@em11.ist.berkeley.edu 'ps aux | grep pretrain_planner'
```

### **View Live Training Logs**
```bash
ssh ghr@em11.ist.berkeley.edu 'cd ~/ghr/Online-Diffusion-Planning/Pretrain && tail -f training_output.log'
```

### **Check GPU Usage**
```bash
ssh ghr@em11.ist.berkeley.edu 'nvidia-smi'
```

## ⚙️ Customization

### **Modify Berkeley Server Details**
Edit the configuration in either script:

```python
# In workflow_automation.py
berkeley_user="ghr"
berkeley_host="em11.ist.berkeley.edu"
```

```bash
# In workflow.sh
BERKELEY_USER="ghr"
BERKELEY_HOST="em11.ist.berkeley.edu"
```

### **Add More Files to Download**
Edit the `files_to_download` list in the Python script or add more `scp` commands in the shell script.

## 🚨 Troubleshooting

### **SSH Connection Issues**
- Ensure your SSH key is added to Berkeley server
- Test connection: `ssh ghr@em11.ist.berkeley.edu`

### **Git Issues**
- Check if you have uncommitted changes: `git status`
- Ensure remote is configured: `git remote -v`

### **Training Not Starting**
- Check if `pretrain_planner.py` exists on Berkeley server
- Verify Python environment on remote server

### **File Download Issues**
- Check if files exist on remote: `ssh ghr@em11.ist.berkeley.edu 'ls -la ~/ghr/Online-Diffusion-Planning/Pretrain/'`
- Ensure sufficient disk space locally

## 💡 Pro Tips

1. **Use descriptive commit messages** to track what changes you made
2. **Run with `--no-wait`** for long training sessions
3. **Check logs periodically** to monitor training progress
4. **Customize file patterns** to download additional file types
5. **Use the Python script** for more complex workflows

## 🔄 Complete Workflow Example

```bash
# 1. Modify your code
vim Pretrain/pretrain_planner.py

# 2. Run automation
python workflow_automation.py -m "Optimized hyperparameters for kitchen environment"

# 3. Wait for completion (or use --no-wait)

# 4. Check results
ls -la Pretrain/Kitchen_High_Planner.pt
ls -la Pretrain/output*.txt
```

## 📞 Support

If you encounter issues:
1. Check the error messages in the script output
2. Verify SSH connectivity to Berkeley server
3. Ensure all dependencies are installed
4. Check file permissions and paths

---

**Happy Training! 🎉**

Your workflow is now fully automated - modify, train, and download with a single command!
