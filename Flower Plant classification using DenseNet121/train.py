#!/usr/bin/env python3
"""
Flower Plant Classification using ResNet50 - Training Pipeline
Extracts high-dimensional ResNet50 deep features from the flower dataset,
trains a calibrated multi-class classifier with stratified cross-validation/evaluation,
and exports the deployment artifact for the Streamlit web application and CLI inference.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Tuple

# IMPORTANT: torch must be imported before other scientific libraries on Windows
import torch
import torchvision
from torchvision.models import ResNet50_Weights, resnet50

import joblib
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


CLASS_FOLDERS = ("daisy", "dandelion", "roses", "sunflowers", "tulips")
CLASS_LABELS = ("Daisy", "Dandelion", "Rose", "Sunflower", "Tulip")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
SEED = 42


def find_project_dirs() -> Tuple[Path, Path, Path]:
    """Resolve project root, data directory, and cache directory regardless of current working directory."""
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir,
        script_dir / "Flower Plant classification using DenseNet121",
        script_dir.parent,
    ]
    
    project_root = script_dir
    data_dir = None
    for cand in candidates:
        possible_data = cand / "data" / "flower_photos"
        if possible_data.is_dir():
            project_root = cand
            data_dir = possible_data
            break
            
    if data_dir is None:
        raise FileNotFoundError(
            f"Could not find flower_photos in any of: {[str(c / 'data' / 'flower_photos') for c in candidates]}"
        )
        
    cache_dir = project_root / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return project_root, data_dir, cache_dir


def collect_dataset(data_dir: Path, max_per_class: int | None = None) -> Tuple[list[Path], np.ndarray]:
    """Scan dataset directory and collect valid image paths and integer class labels."""
    image_paths: list[Path] = []
    labels: list[int] = []
    
    print("\nScanning dataset classes:")
    for label_idx, folder_name in enumerate(CLASS_FOLDERS):
        folder_path = data_dir / folder_name
        if not folder_path.is_dir():
            raise FileNotFoundError(f"Missing expected class folder: {folder_path}")
            
        found = [
            p for p in sorted(folder_path.glob("*"))
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        ]
        
        if max_per_class and len(found) > max_per_class:
            indices = np.linspace(0, len(found) - 1, max_per_class, dtype=int)
            found = [found[i] for i in indices]
            
        print(f"  - {CLASS_LABELS[label_idx]} ({folder_name}): {len(found)} images")
        image_paths.extend(found)
        labels.extend([label_idx] * len(found))
        
    print(f"Total dataset images: {len(image_paths)}\n")
    return image_paths, np.asarray(labels, dtype=np.int64)


def extract_features(
    model: torch.nn.Module,
    transform,
    paths: list[Path],
    labels: np.ndarray,
    batch_size: int = 32,
    device: torch.device = torch.device("cpu"),
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract 2,048-dim features from the ResNet50 avgpool layer in batches."""
    features_list = []
    valid_labels = []
    total = len(paths)
    t0 = time.time()
    
    print(f"Extracting ResNet50 features (batch_size={batch_size}, device={device})...")
    with torch.inference_mode():
        for start_idx in range(0, total, batch_size):
            end_idx = min(start_idx + batch_size, total)
            batch_paths = paths[start_idx:end_idx]
            batch_target_labels = labels[start_idx:end_idx]
            
            tensors = []
            batch_kept = []
            for path, lbl in zip(batch_paths, batch_target_labels):
                try:
                    with Image.open(path) as img:
                        tensors.append(transform(img.convert("RGB")))
                    batch_kept.append(int(lbl))
                except (OSError, UnidentifiedImageError):
                    continue
                    
            if not tensors:
                continue
                
            batch_tensor = torch.stack(tensors).to(device)
            # Forward through ResNet50 convolutional layers & avgpool
            x = model.conv1(batch_tensor)
            x = model.bn1(x)
            x = model.relu(x)
            x = model.maxpool(x)
            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            x = model.layer4(x)
            pooled = model.avgpool(x).flatten(start_dim=1)
            
            features_list.append(pooled.cpu().numpy())
            valid_labels.extend(batch_kept)
            
            processed = min(start_idx + batch_size, total)
            elapsed = time.time() - t0
            rate = processed / elapsed if elapsed > 0 else 0
            percent = (processed / total) * 100
            print(
                f"\r  Progress: {processed}/{total} ({percent:5.1f}%) | "
                f"Rate: {rate:4.1f} img/s | "
                f"Elapsed: {elapsed:5.1f}s",
                end="",
                flush=True,
            )
            
    print("\nFeature extraction completed successfully.")
    return np.concatenate(features_list, axis=0), np.asarray(valid_labels, dtype=np.int64)


def train_and_evaluate(
    features: np.ndarray,
    labels: np.ndarray,
    c_param: float = 1.0,
) -> Tuple[Any, float, np.ndarray, np.ndarray, np.ndarray, str]:
    """Train the classifier pipeline and return metrics and classification report."""
    train_x, test_x, train_y, test_y = train_test_split(
        features,
        labels,
        test_size=0.20,
        random_state=SEED,
        stratify=labels,
    )
    
    print(f"\nTraining dataset: {len(train_y)} images (80%)")
    print(f"Testing holdout:  {len(test_y)} images (20%)")
    
    print("\nFitting StandardScaler + LogisticRegression (ResNet50 feature probe)...")
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c_param,
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=SEED,
        ),
    )
    t_start = time.time()
    clf.fit(train_x, train_y)
    fit_time = time.time() - t_start
    print(f"Classifier fitting complete in {fit_time:.2f}s.")
    
    test_preds = clf.predict(test_x)
    accuracy = float(accuracy_score(test_y, test_preds))
    conf_mat = confusion_matrix(test_y, test_preds, labels=np.arange(len(CLASS_LABELS)))
    report = classification_report(
        test_y,
        test_preds,
        target_names=CLASS_LABELS,
        digits=4,
    )
    
    return clf, accuracy, conf_mat, test_y, test_preds, report


def main():
    parser = argparse.ArgumentParser(description="Train ResNet50 flower classification model")
    parser.add_argument("--batch-size", type=int, default=32, help="Inference batch size for feature extraction")
    parser.add_argument("--max-per-class", type=int, default=None, help="Optional max images per class")
    parser.add_argument("--c", type=float, default=1.0, help="Logistic regression L2 regularization inverse C")
    parser.add_argument("--force-extract", action="store_true", help="Force re-extraction even if cache exists")
    args = parser.parse_args()
    
    print("=" * 65)
    print("  FLOWER PLANT CLASSIFICATION USING RESNET50 - TRAINING")
    print("=" * 65)
    
    project_root, data_dir, cache_dir = find_project_dirs()
    print(f"Project root: {project_root}")
    print(f"Dataset dir:  {data_dir}")
    print(f"Cache dir:    {cache_dir}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:       {device}")
    
    # 1. Load pretrained ResNet50
    print("\nLoading pretrained Torchvision ResNet50 model...")
    weights = ResNet50_Weights.DEFAULT
    resnet = resnet50(weights=weights).to(device)
    resnet.eval()
    for p in resnet.parameters():
        p.requires_grad = False
    transform = weights.transforms()
    
    # 2. Dataset collection
    image_paths, labels = collect_dataset(data_dir, max_per_class=args.max_per_class)
    
    # 3. Check for cached full features
    cache_tag = "full" if args.max_per_class is None else f"limit_{args.max_per_class}"
    feature_cache_path = cache_dir / f"features_resnet50_{cache_tag}.npz"
    
    features = None
    if feature_cache_path.is_file() and not args.force_extract:
        try:
            print(f"Loading cached features from {feature_cache_path.name}...")
            cached = np.load(feature_cache_path)
            if cached["features"].shape[0] == len(image_paths) and cached["features"].shape[1] == 2048:
                features = cached["features"]
                labels = cached["labels"]
                print(f"Successfully loaded cached features: shape {features.shape}")
        except Exception as e:
            print(f"Cache read error: {e}. Re-extracting...")
            features = None
            
    if features is None:
        features, labels = extract_features(
            model=resnet,
            transform=transform,
            paths=image_paths,
            labels=labels,
            batch_size=args.batch_size,
            device=device,
        )
        print(f"Saving extracted features to {feature_cache_path}...")
        np.savez_compressed(
            feature_cache_path,
            features=features,
            labels=labels,
            feature_dim=np.array(2048),
        )
        
    # 4. Train classifier and evaluate
    clf, accuracy, conf_mat, test_y, test_preds, report = train_and_evaluate(
        features, labels, c_param=args.c
    )
    
    print("\n" + "=" * 65)
    print(f"  EVALUATION RESULTS (Held-Out Test Set): ACCURACY = {accuracy * 100:.2f}%")
    print("=" * 65)
    print("\nClassification Report:\n")
    print(report)
    
    print("Confusion Matrix (rows: True, cols: Predicted):")
    conf_df = pd.DataFrame(conf_mat, index=CLASS_LABELS, columns=CLASS_LABELS)
    print(conf_df.to_string())
    print()
    
    # 5. Export artifact for Streamlit & CLI
    model_export_path = cache_dir / "flower_classifier_resnet50.joblib"
    artifact = {
        "backbone": "resnet50",
        "classifier": clf,
        "class_labels": list(CLASS_FOLDERS),
        "test_accuracy": float(accuracy),
        "confusion_matrix": conf_mat.tolist(),
        "test_count": int(len(test_y)),
        "image_count": int(len(labels)),
        "feature_dim": 2048,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    joblib.dump(artifact, model_export_path, compress=3)
    print(f"Saved deployment artifact: {model_export_path}")
    print(f"File size: {os.path.getsize(model_export_path) / 1024:.1f} KB")
    print("\nResNet50 training pipeline finished successfully!")


if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    main()
