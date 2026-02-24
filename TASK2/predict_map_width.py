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
from map_width_regressor import MapWidthRegressor, estimate_magnetic_width
from common import load_and_preprocess, fill_nan_with_median


def preprocess_image(image_path, target_size=(224, 224)):
    tensor = load_and_preprocess(image_path, target_size)
    return tensor.unsqueeze(0)  # add batch dimension


def predict(model_path, image_path, device='cpu'):
    model = MapWidthRegressor(num_channels=4)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    image_tensor = preprocess_image(image_path).to(device)

    with torch.no_grad():
        width_pred = model(image_tensor).item()

    # Also compute FWHM reference for comparison
    img = np.load(image_path, allow_pickle=True)['data'].astype(np.float32)
    img = fill_nan_with_median(img)
    fwhm_ref = estimate_magnetic_width(img)

    return width_pred, fwhm_ref


def main():
    parser = argparse.ArgumentParser(description='Predict magnetic map width')
    parser.add_argument('--input',  type=str, required=True,
                        help='Path to .npz file')
    parser.add_argument('--model',  type=str, default='best_map_width_regressor.pth',
                        help='Path to trained model')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    width_pred, fwhm_ref = predict(args.model, args.input, args.device)

    print(f"\n{'='*50}")
    print("MAP WIDTH PREDICTION")
    print(f"{'='*50}")
    print(f"Predicted width  : {width_pred:.2f} m")
    if fwhm_ref is not None:
        print(f"FWHM reference   : {fwhm_ref:.2f} m")
        print(f"Difference       : {abs(width_pred - fwhm_ref):.2f} m")
    print(f"Valid range      : 5 – 80 m")

    if width_pred < 5 or width_pred > 80:
        print("⚠️  Prediction out of expected range [5, 80 m]")
    else:
        print("✓ Prediction within valid range")


if __name__ == "__main__":
    main()
