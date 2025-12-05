import torch
import torch.nn as nn
from pathlib import Path



def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_samples = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        batch_size = data.y.size(0)
        #print(f"Train batch size = {batch_size}")

        out = model(data)
        loss = criterion(out, data.y)
        # total_loss += loss.item()
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    return total_loss / total_samples


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for data in loader:
            data = data.to(device)

            batch_size = data.y.size(0)
            # print(f"Validate batch size = {batch_size}")

            out = model(data)
            loss = criterion(out, data.y)
            # total_loss += loss.item()
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples


def train_model(model, train_loader, val_loader, epochs, lr, device,
                save_path, max_patience=20, log_interval=10):
    """Full training loop with early stopping"""

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10    #, verbose=True
    )
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        scheduler.step(val_loss)

        last_learning_rate = scheduler.get_last_lr()[0]
        print(f"Current LR: {last_learning_rate:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1

        if (epoch + 1) % log_interval == 0:
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Patience: {patience_counter}/{max_patience}")

        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Load best model
    model.load_state_dict(torch.load(save_path))
    return history
