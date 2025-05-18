import open_clip
import torch
import numpy as np
from PIL import Image
import albumentations as A
import joblib
import os
from collections import Counter

# Load OpenCLIP model with specified weights
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

try:
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name="ViT-L-14",
        pretrained="laion2b_s32b_b82k",  # OpenCLIP weights
        device=device
    )
    print("CLIP model loaded successfully")
except Exception as e:
    print(f"Error loading CLIP model: {str(e)}")
    raise

# Path to test folder where the models are located
test_folder = "/Users/atharvabadkas/Coding /CLIP/v2_testing_pipeline/test"

# Load models from test folder
try:
    classifier = joblib.load(os.path.join(test_folder, "prep_label_classifier.pkl"))
    label_encoder = joblib.load(os.path.join(test_folder, "prep_label_encoder.pkl"))
    print("Classifier and label encoder loaded successfully")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    print(f"Available classes: {', '.join(label_encoder.classes_)}")
except Exception as e:
    print(f"Error loading models: {str(e)}")
    raise

# Simplified augmentation for prediction
augment = A.Compose([
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Resize(224, 224)  # Simple resize to CLIP input size
])

def predict_ingredient(image_path, n_aug=5, confidence_threshold=0.05, debug=True):
    """
    Predict ingredient with ensemble approach and confidence threshold
    
    Args:
        image_path: Path to the image file
        n_aug: Number of augmentations for test-time augmentation
        confidence_threshold: Minimum confidence required for a valid prediction
        debug: Whether to print debug information
        
    Returns:
        tuple: (predicted_class, confidence) or ("unknown", confidence) if below threshold
    """
    try:
        # Input validation
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        if debug:
            print(f"Processing image: {image_path}")
            
        # Load image
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        
        # Check image dimensions
        if debug:
            print(f"Image shape: {image_np.shape}")
            
        if len(image_np.shape) != 3 or image_np.shape[2] != 3:
            if debug:
                print(f"Warning: Unusual image shape {image_np.shape}, attempting to fix")
            # Try to fix the image
            if len(image_np.shape) == 2:  # Grayscale
                image_np = np.stack([image_np, image_np, image_np], axis=2)
            elif len(image_np.shape) == 3 and image_np.shape[2] == 1:  # Single channel
                image_np = np.concatenate([image_np, image_np, image_np], axis=2)
            elif len(image_np.shape) == 3 and image_np.shape[2] == 4:  # RGBA
                image_np = image_np[:, :, :3]
                
        # Ensure image is large enough
        if image_np.shape[0] < 10 or image_np.shape[1] < 10:
            raise ValueError(f"Image too small: {image_np.shape}")
            
        scores = []
        predictions = []
        
        # Multiple augmentation predictions
        for i in range(n_aug):
            try:
                # Apply augmentation
                augmented = augment(image=image_np)["image"]
                
                # Convert to PIL and preprocess for CLIP
                pil_image = Image.fromarray(augmented)
                
                # Use CLIP's preprocess function directly
                preprocessed = preprocess(pil_image).unsqueeze(0).to(device)
                
                # Generate embedding
                with torch.no_grad():
                    features = model.encode_image(preprocessed).cpu().numpy()
                
                # Reshape if needed
                if features.shape[0] == 1 and len(features.shape) > 2:
                    features = features.reshape(1, -1)
                
                # Check for NaN values
                if np.isnan(features).any():
                    if debug:
                        print(f"Warning: NaN values in features for augmentation {i}, skipping")
                    continue
                
                # Get prediction probabilities
                proba = classifier.predict_proba(features)
                scores.append(proba)
                pred_idx = np.argmax(proba)
                predictions.append(pred_idx)
                
                if debug:
                    print(f"Augmentation {i}: shape={features.shape}, max_prob={np.max(proba):.4f}")
                
            except Exception as e:
                if debug:
                    print(f"Error during augmentation {i}: {str(e)}")
                continue
        
        # Check if we have any valid predictions
        if not scores:
            if debug:
                print("No valid predictions generated")
            return None, 0.0
        
        # Ensemble decision making
        avg_proba = np.mean(scores, axis=0)
        max_idx = np.argmax(avg_proba)
        max_confidence = avg_proba[0][max_idx]
        
        # Get most common prediction (majority voting)
        prediction_counts = Counter(predictions)
        most_common_pred, count = prediction_counts.most_common(1)[0]
        
        # Calculate agreement percentage
        agreement = count / len(predictions)
        
        # Use both confidence and agreement for decision
        if debug:
            print(f"Max confidence: {max_confidence:.4f}, Agreement: {agreement:.2%}")
        
        # Only return unknown if both confidence and agreement are low
        if max_confidence < confidence_threshold and agreement < 0.6:
            return "unknown", max_confidence
        
        # Use the class with highest confidence
        predicted_class = label_encoder.inverse_transform([max_idx])[0]
        
        # Get top 3 predictions and their confidences
        top3_indices = np.argsort(avg_proba[0])[-3:][::-1]
        top3_classes = label_encoder.inverse_transform(top3_indices)
        top3_confidences = avg_proba[0][top3_indices]
        
        # Print top 3 predictions
        if debug:
            print("\nTop 3 predictions:")
            for cls, conf in zip(top3_classes, top3_confidences):
                print(f"{cls}: {conf:.2%}")
        
        return predicted_class, max_confidence
    
    except Exception as e:
        if debug:
            print(f"Error during prediction: {str(e)}")
            import traceback
            print(traceback.format_exc())
        return None, 0.0

def batch_predict_ingredients(image_paths, n_aug=5, confidence_threshold=0.05):
    """
    Predict ingredients for multiple images
    
    Args:
        image_paths: List of paths to image files
        n_aug: Number of augmentations for test-time augmentation
        confidence_threshold: Minimum confidence required for a valid prediction
        
    Returns:
        list: List of (predicted_class, confidence) tuples
    """
    results = []
    for image_path in image_paths:
        prediction = predict_ingredient(image_path, n_aug, confidence_threshold)
        results.append(prediction)
    return results

if __name__ == "__main__":
    # Example usage
    image_path = "/Users/atharvabadkas/Coding /CLIP/v2_testing_pipeline/test/DT20250418_TM071641_MCCCBA97012D04_WT1938_TC40_TX34_RN837_DW1803.jpg"
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Warning: Test image not found at {image_path}")
        # Try to find any image file to test with
        test_dir = os.path.dirname(image_path)
        if os.path.exists(test_dir) and os.path.isdir(test_dir):
            image_files = [f for f in os.listdir(test_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if image_files:
                image_path = os.path.join(test_dir, image_files[0])
                print(f"Using alternative test image: {image_path}")
    
    predicted_class, confidence = predict_ingredient(image_path, n_aug=10, confidence_threshold=0.05)
    
    if predicted_class is None:
        print("Failed to make prediction")
    elif predicted_class == "unknown":
        print(f"Prediction uncertain (confidence: {confidence:.2%})")
    else:
        print(f"Predicted: {predicted_class} | Confidence: {confidence:.2%}")