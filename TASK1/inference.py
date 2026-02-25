"""
Script de prédiction pour détecter la présence de conduites
Usage: python predict_pipeline_presence.py --input <file.npz>
"""

import glob
import torch
import numpy as np
import argparse
from scipy.ndimage import zoom
from tqdm import tqdm
from train import PipelinePresenceClassifier

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Prétraite une image .npz pour la prédiction
    """
    data = np.load(image_path, allow_pickle=True)
    image = data['data']  # Shape: (H, W, 4)

    # Convertir en float32 si nécessaire (float16 n'est pas supporté par scipy.zoom)
    if image.dtype == np.float16:
        image = image.astype(np.float32)

    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)  # Remplacer NaN/Inf par des valeurs spécifiques

    # Redimensionner l'image
    image = _resize_image(image)

    # Normaliser les données (par canal)
    image = _normalize_channels(image)

    # Convertir en tensor (H, W, C) -> (C, H, W)
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
    image_tensor = image_tensor.unsqueeze(0)  # Ajouter batch dimension
    
    return image_tensor



def _resize_image(image):
    """Redimensionne l'image à la taille cible"""
    from scipy.ndimage import zoom


    h, w, c = image.shape
    target_h, target_w = 224, 224  # Taille cible fixe

    # Calculer les facteurs de zoom
    zoom_h = target_h / h
    zoom_w = target_w / w

    # Redimensionner chaque canal
    resized = zoom(image, (zoom_h, zoom_w, 1), order=1)

    return resized

def _normalize_channels(image):
    """Normalise chaque canal indépendamment"""
    normalized = np.zeros_like(image)

    for c in range(image.shape[2]):
        channel = image[:, :, c]
        mean = np.mean(channel)
        std = np.std(channel)

        if std > 0:
            normalized[:, :, c] = (channel - mean) / std
        else:
            normalized[:, :, c] = channel - mean

    return normalized

        # C. Gestion cas limite (Image trop petite)
        if h < PATCH_SIZE or w < PATCH_SIZE:
             # Padding simple avec des zéros
             pad_h = max(0, PATCH_SIZE - h)
             pad_w = max(0, PATCH_SIZE - w)
             img = np.pad(img, ((0,0), (0, pad_h), (0, pad_w)), mode='constant')
             h, w = img.shape[1], img.shape[2]

        # D. Trouver le "Point Chaud" (Sniper) sur le canal Norme (Index 3)
        norm_map = img[3, :, :]
        flat_idx = np.argmax(norm_map)
        max_y, max_x = np.unravel_index(flat_idx, norm_map.shape)
        
        # E. Calcul du crop
        half = PATCH_SIZE // 2
        top = max(0, max_y - half)
        left = max(0, max_x - half)
        
        # Ajustement bords
        if top + PATCH_SIZE > h: top = h - PATCH_SIZE
        if left + PATCH_SIZE > w: left = w - PATCH_SIZE
        
        patch = img[:, top:top+PATCH_SIZE, left:left+PATCH_SIZE]
        
        # F. Normalisation (Z-Score) - INDISPENSABLE
        norm_patch = np.zeros_like(patch)
        for i in range(c):
            std = np.std(patch[i])
            if std < 1e-6: std = 1.0
            norm_patch[i] = (patch[i] - np.mean(patch[i])) / std
            
        return torch.from_numpy(norm_patch).float(), True

    except Exception as e:
        print(f"Erreur lecture {Path(npz_path).name}: {e}")
        # Retourne un tenseur vide en cas d'erreur pour ne pas planter le batch
        return torch.zeros((4, PATCH_SIZE, PATCH_SIZE)), False

# ==========================================
# 3. DATASET D'INFÉRENCE
# ==========================================
class InferenceDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        tensor, success = get_sniper_patch(path)
        return tensor, str(path), success

# ==========================================
# 4. MAIN
# ==========================================
if __name__ == "__main__":
    print(f"Utilisation du device : {DEVICE}")

    # 1. Lister les fichiers
    all_files = list(Path(INPUT_FOLDER).glob("*.npz"))
    if not all_files:
        print(f"❌ Aucun fichier .npz trouvé dans {INPUT_FOLDER}")
        exit()
    print(f"📂 Fichiers à traiter : {len(all_files)}")

    # 2. Charger le modèle (Architecture importée de train.py)
    print("🧠 Chargement du modèle...")
    model = PipelinePresenceClassifier(num_channels=4).to(DEVICE)
    
    Returns:
        probability: Probabilité de présence de conduite (0-1)
        prediction: 0 (absence) ou 1 (présence)
    """
    # Charger le modèle
    model = PipelinePresenceClassifier()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # 3. Dataloader
    dataset = InferenceDataset(all_files)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    results = []

    # 4. Inférence
    print("🚀 Démarrage des prédictions...")
    with torch.no_grad():
        for images, paths, successes in tqdm(loader):
            images = images.to(DEVICE)
            
            # Prédiction
            outputs = model(images)
            probs = outputs.cpu().numpy().flatten()
            
            # Stockage
            for i, prob in enumerate(probs):
                if not successes[i]: # Si le fichier était corrompu
                    results.append({
                        "filename": Path(paths[i]).name,
                        "probability": 0.0,
                        "prediction": 0,
                        "status": "ERROR_READING_FILE"
                    })
                    continue

                pred_class = 1 if prob > 0.5 else 0
                label = "Pipe" if pred_class == 1 else "No Pipe"
                
                results.append({
                    "filename": Path(paths[i]).name,
                    "probability": round(float(prob), 4),
                    "prediction": pred_class,
                    "status": label
                })

def main():
    # parser = argparse.ArgumentParser(description='Predict pipeline presence in magnetic images')
    # parser.add_argument('--input', type=str, required=True, help='Path to input .npz file')
    # parser.add_argument('--model', type=str, default='best_pipeline_classifier.pth', 
    #                     help='Path to trained model')
    # parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
    #                     help='Device to use (cuda/cpu)')
    
    # args = parser.parse_args()
    
    # # Prédiction
    # print(f"Loading model from: {args.model}")
    # print(f"Analyzing image: {args.input}")
    no_pipe_predictions = 0
    pipe_prediction= 0
    for file in tqdm(glob.glob("real_data/*.npz"), desc="Analyzing images"):
        # print(f"Analyzing image: {file}")
        
        probability, prediction = predict("V2_pipeline_classifier_epoch2.pth", file, "cuda")
        if prediction == 0: 
            no_pipe_predictions += 1
        elif prediction == 1:
            pipe_prediction += 1

        #     # Afficher les résultats
        # print(f"\n{'='*50}")
        # print("PREDICTION RESULTS")
        # print(f"{'='*50}")
        # print(f"Probability of pipeline presence: {probability:.4f} ({probability*100:.2f}%)")
        # print(f"Prediction: {'PIPELINE DETECTED' if prediction == 1 else 'NO PIPELINE'}")
        # print(f"Confidence: {max(probability, 1-probability)*100:.2f}%")
        
        # if prediction == 0:
        #     print("\n⚠️  No pipeline detected in this image")
        # else:
        #     print("\n✓ Pipeline presence confirmed")

    
    print(f"\nTotal images analyzed: {no_pipe_predictions + pipe_prediction}")
    print(f"Predicted NO PIPELINE: {no_pipe_predictions} images")
    print(f"Predicted PIPELINE DETECTED: {pipe_prediction} images")


if __name__ == "__main__":
    main()
