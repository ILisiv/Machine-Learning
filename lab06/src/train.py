import os
import json
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloader
from model import MLPModel

def load_config(config_path="configs/train_config.yaml"):
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {
        "experiment_name": "Lab6_Training_Pipeline",
        "seed": 42,
        "batch_size": 64,
        "num_workers": 2,
        "epochs": 10,
        "lr": 0.01,
        "optimizer": "sgd_momentum",
        "scheduler": "StepLR",
        "checkpoint_dir": "results/checkpoints"
    }

def train_pipeline(resume_epoch=None):
    config = load_config()
    torch.manual_seed(config["seed"])
    
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    dataloader = get_dataloader(batch_size=config["batch_size"], num_workers=config["num_workers"])
    model = MLPModel()
    criterion = nn.BCELoss()

    if config["optimizer"] == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    else:
        optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9)

    if config["scheduler"] == "CosineAnnealingLR":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    elif config["scheduler"] == "ReduceLROnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    start_epoch = 0
    if resume_epoch is not None:
        ckpt_path = os.path.join(config["checkpoint_dir"], f"epoch_{resume_epoch}.pt")
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path)
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resumed training from epoch {start_epoch}")

    metrics_log = []

    print("Starting training pipeline...")
    for epoch in range(start_epoch, config["epochs"]):
        model.train()
        running_loss = 0.0
        accumulation_steps = 4  # Gradient accumulation study
        optimizer.zero_grad()

        for i, (inputs, targets) in enumerate(dataloader):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss = loss / accumulation_steps
            loss.backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item() * accumulation_steps

        epoch_loss = running_loss / len(dataloader)
        
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(epoch_loss)
        else:
            scheduler.step()

        print(f"Epoch [{epoch+1}/{config['epochs']}], Loss: {epoch_loss:.4f}")
        metrics_log.append({"epoch": epoch + 1, "loss": epoch_loss})

        # Save checkpoint at epoch 3 and final epoch
        if (epoch + 1) in [3, config["epochs"]]:
            ckpt_path = os.path.join(config["checkpoint_dir"], f"epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict()
            }, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

    # Save training logs to results JSON
    with open("results/training_metrics.json", "w") as f:
        json.dump(metrics_log, f, indent=4)
    print("Training pipeline completed successfully and saved to results/training_metrics.json!")

if __name__ == "__main__":
    train_pipeline()