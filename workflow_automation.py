#!/usr/bin/env python3
"""
Workflow Automation Script for Online Diffusion Planning

This script automates the complete development cycle:
1. Modify code locally
2. Upload to Berkeley server
3. Run pretrain_planner.py
4. Download new/modified files back to local

Usage: python workflow_automation.py [options]
"""

import os
import sys
import time
import subprocess
import argparse
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WorkflowAutomation:
    def __init__(self, berkeley_user="ghr", berkeley_host="em11.ist.berkeley.edu"):
        self.berkeley_user = berkeley_user
        self.berkeley_host = berkeley_host
        self.remote_path = f"~/{berkeley_user}/Online-Diffusion-Planning"
        self.local_path = Path.cwd()
        
    def run_command(self, command, description="", check=True):
        """Run a shell command and log the result"""
        logger.info(f"Running: {description}")
        logger.debug(f"Command: {command}")
        
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True, check=check)
            if result.stdout:
                logger.info(f"Output: {result.stdout.strip()}")
            if result.stderr:
                logger.warning(f"Stderr: {result.stderr.strip()}")
            return result
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {e}")
            if check:
                raise
            return e
    
    def git_status(self):
        """Check git status and show what will be committed"""
        logger.info("Checking git status...")
        result = self.run_command("git status", "Git status check")
        return result
    
    def git_commit_and_push(self, commit_message="Auto-commit from workflow automation"):
        """Commit and push changes to GitHub"""
        logger.info("Committing and pushing changes...")
        
        # Add all changes
        self.run_command("git add .", "Adding all changes")
        
        # Commit
        self.run_command(f'git commit -m "{commit_message}"', "Committing changes")
        
        # Push to GitHub
        self.run_command("git push origin main", "Pushing to GitHub")
        
        logger.info("Successfully pushed to GitHub")
    
    def update_berkeley_server(self):
        """Update Berkeley server repository from GitHub"""
        logger.info("Updating Berkeley server repository...")
        
        command = f'ssh {self.berkeley_user}@{self.berkeley_host} "cd {self.remote_path} && git pull origin main"'
        result = self.run_command(command, "Updating Berkeley server")
        
        if result.returncode == 0:
            logger.info("Berkeley server updated successfully")
        else:
            logger.error("Failed to update Berkeley server")
            raise RuntimeError("Berkeley server update failed")
    
    def run_pretrain_planner(self, wait_for_completion=True):
        """Run pretrain_planner.py on Berkeley server"""
        logger.info("Starting pretrain_planner.py on Berkeley server...")
        
        # Start the training process
        command = f'ssh {self.berkeley_user}@{self.berkeley_host} "cd {self.remote_path}/Pretrain && nohup python pretrain_planner.py > training_output.log 2>&1 & echo $!"'
        result = self.run_command(command, "Starting pretrain_planner.py")
        
        if result.returncode == 0:
            pid = result.stdout.strip()
            logger.info(f"Training started with PID: {pid}")
            
            if wait_for_completion:
                logger.info("Waiting for training to complete...")
                self.wait_for_training_completion(pid)
            else:
                logger.info("Training started in background. Check status with: ssh ghr@em11.ist.berkeley.edu 'ps aux | grep pretrain_planner'")
        else:
            logger.error("Failed to start training")
            raise RuntimeError("Training start failed")
    
    def wait_for_training_completion(self, pid):
        """Wait for training process to complete"""
        logger.info("Monitoring training progress...")
        
        while True:
            # Check if process is still running
            command = f'ssh {self.berkeley_user}@{self.berkeley_host} "ps -p {pid} > /dev/null 2>&1 && echo running || echo finished"'
            result = self.run_command(command, "Checking training status", check=False)
            
            if "finished" in result.stdout:
                logger.info("Training completed!")
                break
            elif "running" in result.stdout:
                logger.info("Training still running... waiting 30 seconds")
                time.sleep(30)
            else:
                logger.warning("Could not determine training status, waiting...")
                time.sleep(30)
    
    def download_new_files(self):
        """Download new and modified files from Berkeley server"""
        logger.info("Downloading new and modified files from Berkeley server...")
        
        # Create a list of files to download
        files_to_download = [
            "Pretrain/Kitchen_High_Planner.pt",
            "Pretrain/Kitchen_High_Planner_stats.pkl", 
            "Pretrain/output.txt",
            "Pretrain/output_*.txt",
            "*.pkl",
            "*.pt",
            "*.pth"
        ]
        
        for file_pattern in files_to_download:
            try:
                # Use rsync to download files (more reliable than scp for multiple files)
                command = f'rsync -avz --progress {self.berkeley_user}@{self.berkeley_host}:{self.remote_path}/{file_pattern} ./'
                result = self.run_command(command, f"Downloading {file_pattern}", check=False)
                
                if result.returncode == 0:
                    logger.info(f"Successfully downloaded {file_pattern}")
                else:
                    logger.warning(f"Some files in {file_pattern} may not exist or failed to download")
            except Exception as e:
                logger.warning(f"Failed to download {file_pattern}: {e}")
        
        logger.info("File download completed")
    
    def show_training_logs(self):
        """Show recent training logs from Berkeley server"""
        logger.info("Fetching recent training logs...")
        
        command = f'ssh {self.berkeley_user}@{self.berkeley_host} "cd {self.remote_path}/Pretrain && tail -50 training_output.log"'
        result = self.run_command(command, "Fetching training logs", check=False)
        
        if result.returncode == 0 and result.stdout:
            print("\n" + "="*80)
            print("RECENT TRAINING LOGS:")
            print("="*80)
            print(result.stdout)
            print("="*80 + "\n")
        else:
            logger.warning("No training logs found or failed to fetch")
    
    def run_full_workflow(self, commit_message="Auto-commit from workflow automation", wait_for_training=True):
        """Run the complete workflow"""
        logger.info("Starting full workflow automation...")
        
        try:
            # Step 1: Check git status
            self.git_status()
            
            # Step 2: Commit and push to GitHub
            self.git_commit_and_push(commit_message)
            
            # Step 3: Update Berkeley server
            self.update_berkeley_server()
            
            # Step 4: Run training
            self.run_pretrain_planner(wait_for_completion=wait_for_training)
            
            # Step 5: Download results
            self.download_new_files()
            
            # Step 6: Show training logs
            self.show_training_logs()
            
            logger.info("Full workflow completed successfully! 🎉")
            
        except Exception as e:
            logger.error(f"Workflow failed: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description="Workflow Automation for Online Diffusion Planning")
    parser.add_argument("--commit-message", "-m", default="Auto-commit from workflow automation",
                       help="Commit message for git")
    parser.add_argument("--no-wait", action="store_true",
                       help="Don't wait for training to complete (run in background)")
    parser.add_argument("--berkeley-user", default="ghr",
                       help="Berkeley server username")
    parser.add_argument("--berkeley-host", default="em11.ist.berkeley.edu",
                       help="Berkeley server hostname")
    
    args = parser.parse_args()
    
    # Create workflow automation instance
    workflow = WorkflowAutomation(args.berkeley_user, args.berkeley_host)
    
    try:
        # Run the full workflow
        workflow.run_full_workflow(
            commit_message=args.commit_message,
            wait_for_training=not args.no_wait
        )
    except KeyboardInterrupt:
        logger.info("Workflow interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
