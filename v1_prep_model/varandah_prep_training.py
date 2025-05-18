import os
import open_clip
import torch
import numpy as np
from PIL import Image
import albumentations as A
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import joblib
import tqdm  

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms(
    model_name="ViT-L-14",
    pretrained="laion2b_s32b_b82k",  # OpenCLIP weights
    device=device
)

# Enhanced Augmentation Pipeline
augment = A.Compose([
    A.RandomRotate90(),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussianBlur(p=0.2),
    A.ColorJitter(p=0.2),
])

def generate_embeddings(dataset_path):
    train_embeddings = []
    train_labels = []
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset folder not found: {dataset_path}")
    
    for cls_folder in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, cls_folder)
        if not os.path.isdir(class_path):
            continue
        
        cls_name = cls_folder.replace("_images", "")
        # Collect valid image files first
        img_files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
        
        # Add progress bar for this class
        for img_file in tqdm.tqdm(img_files, desc=f"Class: {cls_name}", leave=False):
            image_path = os.path.join(class_path, img_file)
                
            # Generate 5 augmented versions per image
            for _ in range(5):
                try:
                    image = np.array(Image.open(image_path).convert("RGB"))
                    augmented = augment(image=image)["image"]
                    pil_image = Image.fromarray(augmented)
                    preprocessed = preprocess(pil_image).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        embedding = model.encode_image(preprocessed).cpu().numpy()
                    
                    train_embeddings.append(embedding)
                    train_labels.append(cls_name)
                except Exception as e:
                    print(f"Skipping {image_path}: {str(e)}")
    
    if not train_embeddings:
        raise ValueError("No valid images found in the dataset folder.")
    
    return np.vstack(train_embeddings), np.array(train_labels)

if __name__ == "__main__":
    dataset_path = "/Users/atharvabadkas/Coding /myapp/verandah_prep/varandah_prep_ingredients"  # Update this path
    
    try:
        embeddings, labels = generate_embeddings(dataset_path)
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)
        
        classifier = SVC(kernel='linear', probability=True, C=0.1)
        classifier.fit(embeddings, encoded_labels)
        
        joblib.dump(classifier, "prep_label_classifier.pkl")
        joblib.dump(label_encoder, "prep_label_encoder.pkl")
        print("Training completed. Models saved.")
        
    except Exception as e:
        print(f"Training failed: {str(e)}")