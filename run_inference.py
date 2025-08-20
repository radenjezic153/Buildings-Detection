import torch
import pickle
import argparse
import numpy as np
import torch.nn as nn
import segmentation_models_pytorch as smp

# Here is UNetEfficientNetB0 class
class UNetEfficientNetB0(nn.Module):
    def __init__(self, out_channels):
        super(UNetEfficientNetB0, self).__init__()
        
        # Define the model using SMP with EfficientNet-B0 encoder
        self.model = smp.Unet(
            encoder_name="efficientnet-b0",    # Choose EfficientNet-B0 as the encoder
            encoder_weights="imagenet",        # Use pre-trained ImageNet weights
            in_channels=3,                     # Input channels (3 for RGB images)
            classes=out_channels               # Output channels (e.g., 1 for binary segmentation)
        )

    def forward(self, x):
        # Forward pass through the U-Net model
        return self.model(x)
        
# Define IoU metric function
def calculate_iou(predictions, targets):
    intersection = torch.sum(predictions * targets)
    union = torch.sum(predictions) + torch.sum(targets) - intersection
    iou = intersection / union if union != 0 else torch.tensor(1.0)
    return iou.item()

# Define the main function
def main(model_path, data_path, output_metrics=True):
    # Load the dataset
    with open(data_path, 'rb') as f:
        dataset = pickle.load(f)

    device = torch.device('cpu')
    
    # Unpack dataset (assuming validation data is pre-split)
    validation_dataset = dataset['val']

    # Dataset normalization and preparation
    val_data = [((x['image'] / x['image'].max())[:, :, ::-1], x['mask']) for x in validation_dataset]
    val_data = [(x[0].copy(), x[1].copy()) for x in val_data]
    val_data = [(x[0].transpose(2, 0, 1), x[1]) for x in val_data]

    # Convert numpy arrays to PyTorch tensors
    val_data = [(torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(device),
                 torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device))
                for image, mask in val_data]

    # Load the model
    model = UNetEfficientNetB0(out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set model to evaluation mode
    
    ious = []
    
    with torch.no_grad():  # Disable gradient computation for inference
        for image, true_mask in val_data:
            
            # Perform inference
            pred_mask = model(image)
            pred_mask = torch.sigmoid(pred_mask)
            pred_mask = (pred_mask > 0.5).float()  # Binarize predictions
            
            # Calculate IoU
            iou = calculate_iou(pred_mask.squeeze(0), true_mask.squeeze(0))  # Remove batch and channel dimensions
            ious.append(iou)
    
    # Print metrics
    mean_iou = sum(ious) / len(ious)
    if output_metrics:
        print(f"Mean IoU of the validation dataset is: {mean_iou:.4f}.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference on a dataset using a trained model.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the model weights file.")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the dataset (pickle file).")
    
    args = parser.parse_args()
    main(args.model_path, args.data_path)
