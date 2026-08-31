"""Hydra entry point for the configured finetuning stage."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from Finetuning.finetune_script2 import main


if __name__ == "__main__":
    main()
