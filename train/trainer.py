import os
import torch
from tqdm import tqdm
import warnings
from config.config import OUTPUT_DIR
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler

from utils.metrics import evaluate_model
warnings.filterwarnings("ignore")

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device,
                patience=7, min_delta=0.001):
    """Enhanced training function with mixed precision and learning rate scheduling."""
    best_val_f1 = 0
    train_losses = []
    val_metrics = {'f1': []}

    counter = 0
    early_stop = False

    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    criterion = criterion.to(device)
    is_tpu = str(device).startswith('xla')

    for epoch in range(num_epochs):
        if early_stop:
            print(f"\nEarly stopping triggered after epoch {epoch}")
            break

        model.train()
        running_loss = 0.0

        if is_tpu:
            train_device_loader = pl.ParallelLoader(train_loader, [device]).per_device_loader(device)
        else:
            train_device_loader = train_loader

        for images, labels in tqdm(train_device_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            try:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # Use mixed precision training
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                # Scale gradients and optimize
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                if is_tpu:
                    xm.mark_step()

                running_loss += loss.item()

            except Exception as e:
                print(f"Error in training batch: {str(e)}")
                continue

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {epoch_loss:.4f}')

        val_f1 = evaluate_model(model, val_loader, device, plot_results=False)
        val_metrics['f1'].append(val_f1)

        # Learning rate scheduling
        scheduler.step(val_f1)

        if val_f1 > best_val_f1 + min_delta:
            counter = 0
            best_val_f1 = val_f1

            model_path = os.path.join(OUTPUT_DIR, 'best_model.pth')
            if is_tpu:
                xm.save(model.state_dict(), model_path)
            else:
                torch.save(model.state_dict(), model_path)

            print(f"New best validation F1: {best_val_f1:.4f}")
        else:
            counter += 1
            print(f"EarlyStopping counter: {counter} out of {patience}")

            if counter >= patience:
                early_stop = True
                print("Early stopping triggered")
                break

    return train_losses, val_metrics, epoch + 1
