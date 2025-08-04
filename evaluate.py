import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from PIL import Image
import cv2

from models.ae_unet import AE_UNet
from data.dataset import SegmentationDataset
from utils.metrics import SegmentationMetrics

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset and Dataloader
    # Use the 'val' split for evaluation as a proxy for the test set
    test_dataset = SegmentationDataset(root_dir=args.data_path, dataset_type=args.dataset, split='val')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Model
    model = AE_UNet(n_classes=1).to(device)
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    model.eval()

    # Evaluation
    metrics = SegmentationMetrics(n_classes=2)
    save_dir = args.save_path
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for i, (images, masks) in enumerate(tqdm(test_loader, desc="Evaluating")):
            images = images.to(device)
            
            outputs = model(images)
            preds = torch.sigmoid(outputs) > 0.5
            
            # Update metrics
            metrics.update(preds.cpu().numpy(), masks.cpu().numpy())

            # Save prediction mask
            pred_mask = preds.cpu().numpy().squeeze()
            pred_mask = (pred_mask * 255).astype(np.uint8)
            
            # Get original image name to save prediction with same name
            original_img_name = os.path.basename(test_dataset.image_files[i])
            save_name = original_img_name.replace('.jpg', '.png')
            cv2.imwrite(os.path.join(save_dir, save_name), pred_mask)

    scores = metrics.get_scores()
    print("--- Evaluation Results ---")
    print(f"mIoU: {scores['mIoU']:.4f}")
    print(f"F1-score: {scores['F1']:.4f}")
    print(f"Precision: {scores['Precision']:.4f}")
    print(f"Recall: {scores['Recall']:.4f}")
    print(f"Predictions saved to {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate AE-UNet model.")
    parser.add_argument('--dataset', type=str, required=True, choices=['isic2018', 'kvasir', 'clinicdb'])
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset root directory.')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the trained model checkpoint.')
    parser.add_argument('--save_path', type=str, default='predictions', help='Directory to save predicted masks.')
    
    args = parser.parse_args()
    evaluate(args)