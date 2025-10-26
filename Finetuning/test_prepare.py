from sched import scheduler
import torch.nn as nn
from accelerate import Accelerator
from torch.utils.data import DataLoader
import torch

def main():
    accelerator = Accelerator()
    model = nn.Linear(10, 10)  # Simple model
    optimizer = torch.optim.Adam(model.parameters())
    data = torch.randn(10, 10)
    dataloader = DataLoader(data, batch_size = 2, shuffle = True, num_workers = 0, pin_memory = True, drop_last = True)
    print(f"Process {accelerator.process_index}: Starting prepare...")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 100)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    print(f"Process {accelerator.process_index}: Prepare done!")

if __name__ == "__main__":
    torch.manual_seed(1)
    main()