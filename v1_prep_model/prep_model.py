import open_clip
import torch
import numpy as np
from PIL import Image
import albumentations as A
import joblib
import os

# Load OpenCLIP model with specified weights
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms(
    model_name="ViT-L-14",
    pretrained="laion2b_s32b_b82k",  # OpenCLIP weights
    device=device
)

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load models with correct paths
classifier = joblib.load(os.path.join(current_dir, "prep_label_classifier.pkl"))
label_encoder = joblib.load(os.path.join(current_dir, "prep_label_encoder.pkl"))

# TTA Augmentation
augment = A.Compose([
    A.RandomRotate90(),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomBrightnessContrast(p=0.3),
])

def predict_ingredient(image_path, n_aug=5):
    try:
        image = Image.open(image_path).convert("RGB")
        scores = []
        
        for _ in range(n_aug):
            augmented = augment(image=np.array(image))["image"]
            pil_image = Image.fromarray(augmented)
            
            preprocessed = preprocess(pil_image).unsqueeze(0).to(device)
            with torch.no_grad():
                features = model.encode_image(preprocessed).cpu().numpy()
            
            proba = classifier.predict_proba(features)
            scores.append(proba)
        
        avg_proba = np.mean(scores, axis=0)
        predicted_idx = np.argmax(avg_proba)
        predicted_class = label_encoder.inverse_transform([predicted_idx])[0]
        confidence = np.max(avg_proba)
        
        return predicted_class, confidence
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return None, 0.0

if __name__ == "__main__":
    image_path = "/Users/atharvabadkas/Coding /CLIP/v1_prep_clip_test/verandah_prep/varandah_prep_ingredients/bombay sandwich stuffing/DT20241109_TM074701_MCD83BDA89443C_WT2240_TC39_TX37_RN640.jpg"  # Replace with your image
    predicted_class, confidence = predict_ingredient(image_path)
    print(f"Predicted: {predicted_class} | Confidence: {confidence:.2%}")
