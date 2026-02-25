"""
Script de prédiction de la largeur de carte magnétique
Usage: python predict_map_width.py --input <file.npz>
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import argparse
from map_width_regressor import MapWidthRegressor
from common import load_and_preprocess


def predict(model_path, image_path, device='cpu'):
    model = MapWidthRegressor(num_channels=4)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Load raw to get original spatial dimensions
    img_raw = np.load(image_path, allow_pickle=True)['data']
    orig_h, orig_w = img_raw.shape[:2]

    image_tensor = load_and_preprocess(image_path).unsqueeze(0).to(device)  # (1,C,H,W)

    # Metadata: physical dimensions [H_m, W_m] only — no FWHM leakage
    meta = torch.tensor([[orig_h * 0.2, orig_w * 0.2]],
                        dtype=torch.float32, device=device)

    with torch.no_grad():
        width_pred = model(image_tensor, meta).item()

    return width_pred


def main():
    parser = argparse.ArgumentParser(description='Predict magnetic map width')
    parser.add_argument('--input',  type=str, required=True,
                        help='Path to .npz file')
    parser.add_argument('--model',  type=str, default='best_map_width_regressor.pth',
                        help='Path to trained model')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    width_pred = predict(args.model, args.input, args.device)

    print(f"\n{'='*50}")
    print("MAP WIDTH PREDICTION")
    print(f"{'='*50}")
    print(f"Predicted width  : {width_pred:.2f} m")
    print(f"Valid range      : ~2 – 160 m")

    if width_pred < 2 or width_pred > 160:
        print("⚠️  Prediction out of expected range")
    else:
        print("✓ Prediction within valid range")


if __name__ == "__main__":
    main()
