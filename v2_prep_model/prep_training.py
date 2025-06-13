import os
import open_clip
import torch
import numpy as np
from PIL import Image
import albumentations as A
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import joblib
import tqdm
from collections import defaultdict
import sys

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

try:
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name="ViT-L-14",
        pretrained="laion2b_s32b_b82k",
        device=device
    )
    print("CLIP model loaded successfully")
except Exception as e:
    print(f"Error loading CLIP model: {str(e)}")
    raise

# Simplified augmentation pipeline to avoid errors
augment = A.Compose([
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Resize(224, 224)  # Simple resize to CLIP input size
])

def generate_balanced_embeddings(dataset_path, min_samples_per_class=None, aug_per_image=10, use_all_images=True):
    """
    Generate balanced embeddings by taking equal samples from each class
    and applying multiple augmentations
    
    Args:
        dataset_path: Path to the dataset directory
        min_samples_per_class: Minimum number of samples per class (None = auto-determine)
        aug_per_image: Number of augmentations to generate per image
        use_all_images: Whether to use all available images (True) or limit to min_samples_per_class (False)
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
        
    train_embeddings = []
    train_labels = []
    class_counts = defaultdict(int)
    
    # First, count available images per class
    print("Scanning dataset directory...")
    valid_classes = []
    for cls_folder in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, cls_folder)
        if not os.path.isdir(class_path):
            continue
        
        img_files = [f for f in os.listdir(class_path) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        if img_files:
            class_counts[cls_folder] = len(img_files)
            valid_classes.append(cls_folder)
            print(f"Found {len(img_files)} images in class {cls_folder}")
    
    if not valid_classes:
        raise ValueError("No valid class folders found in the dataset directory")
    
    # Determine minimum samples per class if not specified
    if min_samples_per_class is None:
        min_samples = min(class_counts.values())
        min_samples_per_class = min_samples
    
    print(f"\nTotal classes: {len(valid_classes)}")
    if use_all_images:
        print(f"Using ALL available images per class")
        print(f"Generating {aug_per_image} augmentations per image")
        total_images = sum(class_counts.values())
        print(f"Total images to process: {total_images}")
    else:
        print(f"Using {min_samples_per_class} samples per class")
        print(f"Generating {aug_per_image} augmentations per image")
        total_images = min_samples_per_class * len(valid_classes)
        print(f"Total images to process: {total_images}")
    
    for cls_folder in valid_classes:
        class_path = os.path.join(dataset_path, cls_folder)
        cls_name = cls_folder.replace("_images", "")
        img_files = [f for f in os.listdir(class_path) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        
        # If not using all images, limit to min_samples_per_class
        if not use_all_images and len(img_files) > min_samples_per_class:
            np.random.seed(42)  # For reproducibility
            img_files = np.random.choice(img_files, min_samples_per_class, replace=False)
        
        print(f"\nProcessing class {cls_name}:")
        print(f"- Using {len(img_files)} images")
        print(f"- Generating {aug_per_image} augmentations per image")
        
        class_embeddings = []
        class_labels = []
        
        for img_file in tqdm.tqdm(img_files, desc=f"Processing {cls_name}"):
            image_path = os.path.join(class_path, img_file)
            
            try:
                # Load and verify image
                image = Image.open(image_path).convert("RGB")
                image_np = np.array(image)
                
                if image_np.size == 0 or image_np.shape[0] == 0 or image_np.shape[1] == 0:
                    print(f"Warning: Empty image at {image_path}, skipping")
                    continue
                
                for aug_idx in range(aug_per_image):
                    try:
                        # Apply augmentation
                        augmented = augment(image=image_np)["image"]
                        
                        # Convert to PIL and preprocess for CLIP
                        pil_image = Image.fromarray(augmented)
                        preprocessed = preprocess(pil_image).unsqueeze(0).to(device)
                        
                        # Generate embedding
                        with torch.no_grad():
                            embedding = model.encode_image(preprocessed).cpu().numpy()
                        
                        # Check for NaN values
                        if np.isnan(embedding).any():
                            print(f"Warning: NaN values in embedding for {image_path}, skipping")
                            continue
                            
                        class_embeddings.append(embedding)
                        class_labels.append(cls_name)
                        
                    except Exception as aug_error:
                        print(f"Error during augmentation {aug_idx} of {image_path}: {str(aug_error)}")
                        continue
                    
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                continue
        
        # Add class embeddings to overall dataset
        if class_embeddings:
            train_embeddings.extend(class_embeddings)
            train_labels.extend(class_labels)
            print(f"Added {len(class_embeddings)} embeddings for class {cls_name}")
        else:
            print(f"Warning: No valid embeddings generated for class {cls_name}")
    
    if not train_embeddings:
        raise ValueError("No valid embeddings generated. Check your dataset and image processing pipeline.")
    
    # Convert to numpy arrays
    try:
        embeddings_array = np.vstack([e.reshape(1, -1) for e in train_embeddings])
        labels_array = np.array(train_labels)
        print(f"Successfully created embeddings array with shape {embeddings_array.shape}")
        return embeddings_array, labels_array
    except Exception as e:
        print(f"Error creating final arrays: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Simple direct path to dataset
        dataset_path = "/Users/atharvabadkas/Coding /CLIP/testing_pipeline/Datasets/verandah_prep_ingredients"
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset directory not found at: {dataset_path}")
            
        print(f"Using dataset path: {dataset_path}")
        
        # Generate balanced embeddings
        print("\nGenerating balanced embeddings...")
        embeddings, labels = generate_balanced_embeddings(
            dataset_path,
            min_samples_per_class=None,  # Auto-determine based on smallest class
            aug_per_image=10,  # Changed from 5 to 10
            use_all_images=True  # Use ALL available images
        )
        
        print("\nEmbeddings shape:", embeddings.shape)
        print("Number of labels:", len(labels))
        
        # Encode labels
        print("\nEncoding labels...")
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)
        
        # Compute class weights
        print("\nComputing class weights...")
        unique_classes = np.unique(encoded_labels)
        if len(unique_classes) < 2:
            print("Warning: Only one class found in the dataset. Cannot train classifier.")
            sys.exit(1)
            
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=unique_classes,
            y=encoded_labels
        )
        class_weight_dict = dict(zip(unique_classes, class_weights))
        
        # Train SVM with class weights
        print("\nTraining SVM classifier...")
        classifier = SVC(
            kernel='linear',
            probability=True,
            C=0.1,
            class_weight=class_weight_dict
        )
        classifier.fit(embeddings, encoded_labels)
        
        # Save models
        print("\nSaving models...")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        joblib.dump(classifier, os.path.join(current_dir, "verandah_prep_label_classifier.pkl"))
        joblib.dump(label_encoder, os.path.join(current_dir, "verandah_prep_label_encoder.pkl"))
        
        print("\nTraining completed successfully!")
        print(f"Number of classes: {len(np.unique(labels))}")
        print(f"Total samples: {len(labels)}")
        print("\nClass distribution:")
        for cls in np.unique(labels):
            count = np.sum(labels == cls)
            print(f"{cls}: {count} samples")
        
    except Exception as e:
        print(f"\nTraining failed with error: {str(e)}")
        import traceback
        print("\nFull error traceback:")
        print(traceback.format_exc())
