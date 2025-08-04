import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.ae_unet import AE_UNet
from data.dataset import SegmentationDataset
from utils.metrics import SegmentationMetrics

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0.0
    
    for images, masks in tqdm(dataloader, desc="Training"):
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0.0
    metrics = SegmentationMetrics(n_classes=2) # Binary: background and foreground
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validation"):
            images, masks = images.to(device), masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            epoch_loss += loss.item()
            
            preds = torch.sigmoid(outputs) > 0.5
            metrics.update(preds.cpu().numpy(), masks.cpu().numpy())
            
    val_scores = metrics.get_scores()
    return epoch_loss / len(dataloader), val_scores

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Datasets and Dataloaders
    train_dataset = SegmentationDataset(root_dir=args.data_path, dataset_type=args.dataset, split='train')
    val_dataset = SegmentationDataset(root_dir=args.data_path, dataset_type=args.dataset, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Model
    model = AE_UNet(n_classes=1).to(device)
    
    # Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)
    
    # Training Loop
    best_f1 = 0.0
    os.makedirs('checkpoints', exist_ok=True)
    
    for epoch in range(args.epochs):
        print(f"--- Epoch {epoch+1}/{args.epochs} ---")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_scores = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val mIoU: {val_scores['mIoU']:.4f} | Val F1: {val_scores['F1']:.4f} | Val Precision: {val_scores['Precision']:.4f} | Val Recall: {val_scores['Recall']:.4f}")
        
        # Save best model
        if val_scores['F1'] > best_f1:
            best_f1 = val_scores['F1']
            save_path = f"checkpoints/{args.model_name}_{args.dataset}_best.pth"
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AE-UNet model.")
    parser.add_argument('--dataset', type=str, required=True, choices=['isic2018', 'kvasir', 'clinicdb'], help='Dataset to use.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset root directory.')
    parser.add_argument('--epochs', type=int, default=120, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')
    parser.add_argument('--model_name', type=str, default='ae_unet', help='Name for saving the model.')
    
    args = parser.parse_args()
    main(args)