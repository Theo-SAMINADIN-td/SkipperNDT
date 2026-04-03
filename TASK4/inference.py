"""
Inference Script for Pipe Type Classifier
"""

import argparse
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from train import PipeTypeClassifier, PipeTypeDataset, TARGET_SIZE, BATCH_SIZE, NUM_WORKERS


def load_model(model_path, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = PipeTypeClassifier(num_channels=4).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"✓ Model loaded from {model_path}")
    return model


def predict(model, dataloader, device):
    model.eval()
    predictions = []

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).float().cpu().numpy()
            predictions.extend(preds)

    return np.array(predictions)


def main():
    parser = argparse.ArgumentParser(description="Inference for Pipe Type Classifier")
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to the trained model file"
    )
    parser.add_argument(
        "--input-data", type=str, required=True, help="Path to .npz folder for inference"
    )
    parser.add_argument(
        "--output-file", type=str, required=True, help="Path to save predictions as a .csv file"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = load_model(args.model_path, device)

    # Prepare dataset and dataloader
    file_paths = [
        os.path.join(args.input_data, f)
        for f in os.listdir(args.input_data)
        if f.endswith(".npz")
    ]
    dataset = PipeTypeDataset(file_paths, labels=[0] * len(file_paths), target_size=TARGET_SIZE, augment=False)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda")
    )

    # Perform inference
    print("Performing inference...")
    predictions = predict(model, dataloader, device)

    # Save predictions
    output_path = args.output_file
    with open(output_path, "w") as f:
        f.write("file_name,prediction\n")
        for file_path, pred in zip(file_paths, predictions):
            f.write(f"{os.path.basename(file_path)},{int(pred[0])}\n")

    print(f"✓ Predictions saved to {output_path}")


if __name__ == "__main__":
    main()