#!/usr/bin/env python3
"""
Flower Plant Classification using ResNet50 - Inference & Prediction CLI
Accurately predicts flower species for any input image file using the trained ResNet50 classifier.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import random

# IMPORTANT: torch must be imported before other scientific libraries on Windows
import torch
import torchvision
from torchvision.models import ResNet50_Weights, resnet50

import joblib
import numpy as np
from PIL import Image, UnidentifiedImageError


CLASS_FOLDERS = ("daisy", "dandelion", "roses", "sunflowers", "tulips")
CLASS_LABELS = ("Daisy", "Dandelion", "Rose", "Sunflower", "Tulip")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_artifact_and_data() -> tuple[Path, Path]:
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir,
        script_dir / "Flower Plant classification using DenseNet121",
        script_dir.parent,
    ]
    
    artifact_path = None
    data_dir = None
    for cand in candidates:
        art = cand / ".cache" / "flower_classifier_resnet50.joblib"
        if art.is_file():
            artifact_path = art
            break
            
    for cand in candidates:
        d = cand / "data" / "flower_photos"
        if d.is_dir():
            data_dir = d
            break
            
    if artifact_path is None:
        raise FileNotFoundError(
            "Trained model artifact '.cache/flower_classifier_resnet50.joblib' not found. "
            "Please run 'python train.py' first to train and export the model."
        )
    return artifact_path, data_dir


def load_model(artifact_path: Path):
    """Load ResNet50 feature extractor and fitted classifier pipeline."""
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights).to(DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    transform = weights.transforms()
    
    artifact = joblib.load(artifact_path)
    return model, transform, artifact


def predict_image(
    image_path: Path,
    model: torch.nn.Module,
    transform,
    classifier,
) -> tuple[str, float, list[tuple[str, float]]]:
    """Given an image path, return predicted class name, confidence, and all class probabilities."""
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {image_path}")
        
    try:
        with Image.open(image_path) as img:
            tensor = transform(img.convert("RGB")).unsqueeze(0).to(DEVICE)
    except Exception as e:
        raise ValueError(f"Could not open or process image {image_path}: {e}")
        
    with torch.inference_mode():
        x = model.conv1(tensor)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        features = model.avgpool(x).flatten(start_dim=1)
        
    probabilities = classifier.predict_proba(features.cpu().numpy())[0]
    best_idx = int(np.argmax(probabilities))
    confidence = float(probabilities[best_idx])
    
    ranked_indices = np.argsort(probabilities)[::-1]
    all_probs = [(CLASS_LABELS[i], float(probabilities[i])) for i in ranked_indices]
    
    return CLASS_LABELS[best_idx], confidence, all_probs


def main():
    parser = argparse.ArgumentParser(description="Predict flower species from an image using ResNet50")
    parser.add_argument("image_path", nargs="?", default=None, help="Path to flower photograph (JPG, PNG, WebP)")
    parser.add_argument("--random-sample", action="store_true", help="Pick a random image from the dataset for testing")
    args = parser.parse_args()
    
    print("=" * 60)
    print("  FLOWER PLANT CLASSIFIER - RESNET50 INFERENCE")
    print("=" * 60)
    
    artifact_path, data_dir = find_artifact_and_data()
    
    target_image = None
    if args.image_path:
        target_image = Path(args.image_path)
    elif args.random_sample or args.image_path is None:
        if data_dir is not None:
            all_imgs = [
                p for p in data_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            ]
            if all_imgs:
                target_image = random.choice(all_imgs)
                print(f"No image provided. Selected random dataset sample:\n  {target_image}\n  (Actual folder: {target_image.parent.name})\n")
        if target_image is None:
            print("Please specify an image path: python predict.py <path_to_image>")
            sys.exit(1)
            
    print(f"Target Image:  {target_image.resolve()}")
    print("Loading model and weights...")
    model, transform, artifact = load_model(artifact_path)
    print(f"Loaded classifier trained on {artifact.get('image_count', 'N/A')} images with test accuracy: {artifact.get('test_accuracy', 0):.2%}")
    
    predicted_label, confidence, prob_list = predict_image(
        target_image, model, transform, artifact["classifier"]
    )
    
    print("\n" + "-" * 40)
    print(f"  PREDICTION:  {predicted_label.upper()}")
    print(f"  CONFIDENCE:  {confidence * 100:.2f}%")
    print("-" * 40)
    print("\nClass Probabilities:")
    for flower, prob in prob_list:
        bar_len = int(prob * 25)
        bar = "#" * bar_len + "-" * (25 - bar_len)
        print(f"  {flower:12} : {prob * 100:6.2f}%  [{bar}]")
    print()


if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    main()
