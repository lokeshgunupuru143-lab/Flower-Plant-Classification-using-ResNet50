from __future__ import annotations

import os
import hashlib
import tarfile
from pathlib import Path
from typing import Any

# PyTorch must be imported before other C-extension scientific packages on Windows
import torch
import torchvision
from torchvision.models import ResNet50_Weights, resnet50

import numpy as np
import pandas as pd
import requests
import streamlit as st
import joblib
from PIL import Image, UnidentifiedImageError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
DATA_DIR = DATA_ROOT / "flower_photos"
ARCHIVE_PATH = DATA_ROOT / "flower_photos.tgz"
MODEL_ARTIFACT_PATH = ROOT / ".cache" / "flower_classifier_resnet50.joblib"
DATA_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
CLASS_FOLDERS = ("daisy", "dandelion", "roses", "sunflowers", "tulips")
CLASS_LABELS = ("Daisy", "Dandelion", "Rose", "Sunflower", "Tulip")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
MAX_IMAGES_PER_CLASS = 250
FEATURE_CACHE_DIR = ROOT / ".cache"
FEATURE_CACHE_VERSION = "resnet50-imagenet-v1"
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(
    page_title="Flower Field | ResNet50",
    page_icon="🌼",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');
    :root { --ink: #1d2b24; --leaf: #286449; --moss: #dfe9df; --paper: #f5f6f0; --sun: #e8b54a; --muted: #4d5d53; }
    .stApp { background: var(--paper); color: var(--ink); }
    [data-testid="stHeader"] { background: rgba(245, 246, 240, .9); }
    h1, h2, h3 { color: var(--ink); }
    h1 { font-family: 'Playfair Display', Georgia, serif; font-size: 2.45rem !important; line-height: 1.12 !important; }
    h2, h3, p, label { font-family: 'DM Sans', sans-serif; color: var(--ink); }
    .stMarkdown p { line-height: 1.55; }
    [data-testid="stCaptionContainer"] { color: var(--muted); font-size: .9rem; }
    [data-testid="stSidebar"] { background: #e8eee5; border-right: 1px solid #d5dfd4; }
    [data-testid="stMetric"] { background: #fff; border: 1px solid #dce4da; border-radius: 6px; padding: 14px 16px; }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    .eyebrow { font-family: 'DM Mono', monospace; color: var(--leaf); font-size: .8rem; text-transform: uppercase; }
    .lede { color: var(--muted); font-size: 1.02rem; max-width: 58rem; line-height: 1.55; }
    div.stButton > button[kind="primary"] { background: var(--leaf); border-color: var(--leaf); }
    div.stButton > button[kind="primary"]:hover { background: #1e5038; border-color: #1e5038; }
    [data-testid="stFileUploader"] { background: rgba(255,255,255,.7); border-radius: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def ensure_dataset() -> None:
    """Download and safely extract the public flower_photos archive if needed."""
    if DATA_DIR.is_dir() and all((DATA_DIR / name).is_dir() for name in CLASS_FOLDERS):
        return

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE_PATH.exists():
        temporary_archive = ARCHIVE_PATH.with_suffix(".tgz.part")
        try:
            with requests.get(DATA_URL, stream=True, timeout=(20, 180)) as response:
                response.raise_for_status()
                with temporary_archive.open("wb") as output_file:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            output_file.write(chunk)
            temporary_archive.replace(ARCHIVE_PATH)
        finally:
            temporary_archive.unlink(missing_ok=True)

    destination = DATA_ROOT.resolve()
    with tarfile.open(ARCHIVE_PATH, mode="r:gz") as archive:
        safe_members = []
        for member in archive.getmembers():
            target = (DATA_ROOT / member.name).resolve()
            if target != destination and destination not in target.parents:
                raise ValueError(f"Unsafe path in flower archive: {member.name}")
            if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
                raise ValueError(f"Unsupported entry in flower archive: {member.name}")
            safe_members.append(member)
        archive.extractall(path=DATA_ROOT, members=safe_members)

    missing = [name for name in CLASS_FOLDERS if not (DATA_DIR / name).is_dir()]
    if missing:
        raise FileNotFoundError(f"Dataset is missing expected class folders: {missing}")


def collect_images() -> tuple[list[Path], np.ndarray]:
    image_paths: list[Path] = []
    labels: list[int] = []
    for label, folder_name in enumerate(CLASS_FOLDERS):
        class_paths = [
            path
            for path in sorted((DATA_DIR / folder_name).rglob("*"))
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]
        if len(class_paths) > MAX_IMAGES_PER_CLASS:
            chosen_indices = np.linspace(
                0, len(class_paths) - 1, MAX_IMAGES_PER_CLASS, dtype=int
            )
            class_paths = [class_paths[index] for index in chosen_indices]
        image_paths.extend(class_paths)
        labels.extend([label] * len(class_paths))
    if not image_paths:
        raise RuntimeError(f"No supported images were found in {DATA_DIR}")
    return image_paths, np.asarray(labels, dtype=np.int64)


def dataset_fingerprint(paths: list[Path], weights_name: str) -> str:
    """Fingerprint selected images and weights to invalidate stale feature caches."""
    digest = hashlib.sha256(FEATURE_CACHE_VERSION.encode())
    digest.update(weights_name.encode())
    for path in paths:
        stat = path.stat()
        entry = f"{path.relative_to(DATA_ROOT).as_posix()}|{stat.st_size}|{stat.st_mtime_ns}\n"
        digest.update(entry.encode())
    return digest.hexdigest()


def extract_features(
    model: torch.nn.Module,
    transform: Any,
    paths: list[Path],
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run frozen ResNet50 inference in small batches and return pooled features."""
    features: list[np.ndarray] = []
    kept_labels: list[int] = []
    batch_size = 16

    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            batch_tensors = []
            batch_labels = []
            for path, label in zip(paths[start : start + batch_size], labels[start : start + batch_size]):
                try:
                    with Image.open(path) as image:
                        batch_tensors.append(transform(image.convert("RGB")))
                    batch_labels.append(int(label))
                except (OSError, UnidentifiedImageError):
                    continue

            if not batch_tensors:
                continue

            image_batch = torch.stack(batch_tensors).to(DEVICE)
            feature_map = model.maxpool(model.relu(model.bn1(model.conv1(image_batch))))
            feature_map = model.layer1(feature_map)
            feature_map = model.layer2(feature_map)
            feature_map = model.layer3(feature_map)
            feature_map = model.layer4(feature_map)
            pooled = model.avgpool(feature_map).flatten(start_dim=1)
            features.append(pooled.cpu().numpy())
            kept_labels.extend(batch_labels)

    if not features:
        raise RuntimeError("ResNet50 could not read any dataset images.")
    return np.concatenate(features), np.asarray(kept_labels, dtype=np.int64)


@st.cache_resource(show_spinner=False)
def load_classifier() -> dict[str, Any]:
    """Load the notebook-trained classifier, or create a cached fallback."""
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights).to(DEVICE)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    transform = weights.transforms()

    if MODEL_ARTIFACT_PATH.is_file():
        artifact = joblib.load(MODEL_ARTIFACT_PATH)
        if artifact.get("backbone") != "resnet50":
            raise ValueError(
                "The saved classifier is not a ResNet50 export. Re-run the notebook's Streamlit export cell."
            )
        if tuple(artifact.get("class_labels", ())) != CLASS_FOLDERS:
            raise ValueError(
                "The saved classifier has different class labels. Re-run the notebook's Streamlit export cell."
            )
        return {
            "model": model,
            "transform": transform,
            "classifier": artifact["classifier"],
            "test_accuracy": float(artifact["test_accuracy"]),
            "confusion_matrix": np.asarray(artifact["confusion_matrix"]),
            "test_count": int(artifact["test_count"]),
            "image_count": int(artifact["image_count"]),
        }

    ensure_dataset()
    image_paths, labels = collect_images()
    fingerprint = dataset_fingerprint(image_paths, weights.name)
    cache_path = FEATURE_CACHE_DIR / f"features-{fingerprint}.npz"
    all_features = None
    all_labels = None
    if cache_path.is_file():
        try:
            with np.load(cache_path, allow_pickle=False) as cached_features:
                if cached_features["fingerprint"].item() == fingerprint:
                    all_features = cached_features["features"].copy()
                    all_labels = cached_features["labels"].copy()
        except (OSError, KeyError, ValueError):
            all_features = None
            all_labels = None

    if all_features is None or all_labels is None:
        all_features, all_labels = extract_features(model, transform, image_paths, labels)
        temporary_cache = cache_path.with_suffix(".npz.tmp")
        try:
            FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with temporary_cache.open("wb") as cache_file:
                np.savez_compressed(
                    cache_file,
                    features=all_features,
                    labels=all_labels,
                    fingerprint=np.asarray(fingerprint),
                )
            temporary_cache.replace(cache_path)
        except OSError:
            temporary_cache.unlink(missing_ok=True)

    train_features, test_features, train_targets, test_targets = train_test_split(
        all_features,
        all_labels,
        test_size=0.20,
        random_state=SEED,
        stratify=all_labels,
    )

    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
    )
    classifier.fit(train_features, train_targets)
    predictions = classifier.predict(test_features)

    return {
        "model": model,
        "transform": transform,
        "classifier": classifier,
        "test_accuracy": float(accuracy_score(test_targets, predictions)),
        "confusion_matrix": confusion_matrix(
            test_targets, predictions, labels=np.arange(len(CLASS_LABELS))
        ),
        "test_count": int(len(test_targets)),
        "image_count": int(len(all_labels)),
    }


def predict_image(image: Image.Image, artifacts: dict[str, Any]) -> np.ndarray:
    image_tensor = artifacts["transform"](image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        model = artifacts["model"]
        feature_map = model.maxpool(model.relu(model.bn1(model.conv1(image_tensor))))
        feature_map = model.layer1(feature_map)
        feature_map = model.layer2(feature_map)
        feature_map = model.layer3(feature_map)
        feature_map = model.layer4(feature_map)
        features = model.avgpool(feature_map).flatten(start_dim=1)
    return artifacts["classifier"].predict_proba(features.cpu().numpy())[0]


def get_demo_images() -> list[Path]:
    if not DATA_DIR.is_dir():
        return []
    return [
        path
        for folder_name in CLASS_FOLDERS
        for path in sorted((DATA_DIR / folder_name).glob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]


with st.sidebar:
    st.markdown('<div class="eyebrow">Model card</div>', unsafe_allow_html=True)
    st.subheader("ResNet50")
    st.caption("Frozen ImageNet features · five flower labels")
    st.markdown("---")
    st.markdown("**Classes**")
    st.write("Daisy · Dandelion · Rose · Sunflower · Tulip")
    st.markdown("**Input**")
    st.write("RGB photograph")
    st.markdown("**Dataset**")
    st.caption("TensorFlow flower_photos sample; Flickr image terms remain with their creators.")
    st.markdown("**Model source**")
    st.markdown("[Torchvision ResNet50](https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html)")
    st.caption("Torchvision repository: BSD-3-Clause. Review current pretrained-weight terms before redistribution or commercial use.")
    st.markdown("---")
    st.caption(f"Compute device: {DEVICE}")

st.markdown('<div class="eyebrow">Image recognition · pretrained vision</div>', unsafe_allow_html=True)
st.title("Flower Field")
st.markdown(
    '<p class="lede">Identify one of five flower categories from a photograph. ResNet50 supplies frozen visual features; a lightweight classifier maps them to the flower labels.</p>',
    unsafe_allow_html=True,
)

prediction_tab, evaluation_tab, about_tab = st.tabs(["Identify", "Evaluation", "How it works"])

with prediction_tab:
    st.subheader("Classify a flower image")
    source_mode = st.radio("Image source", ["Upload an image", "Choose a dataset example"], horizontal=True)
    selected_image: Image.Image | None = None
    upload_name = "Uploaded image"

    if source_mode == "Upload an image":
        uploaded_file = st.file_uploader("Choose a JPG, PNG, or WebP image", type=["jpg", "jpeg", "png", "webp"])
        if uploaded_file is not None:
            upload_name = uploaded_file.name
            try:
                selected_image = Image.open(uploaded_file).convert("RGB")
            except (OSError, UnidentifiedImageError):
                st.error("That file could not be opened as an image. Try a JPG or PNG file.")
    else:
        demo_images = get_demo_images()
        if not demo_images:
            st.info("Download the public flower sample set to use a bundled example image.")
            if st.button("Download sample images"):
                try:
                    with st.spinner("Downloading and preparing the flower sample set…"):
                        ensure_dataset()
                    st.rerun()
                except Exception as error:
                    st.error(f"Could not prepare the sample dataset: {error}")
        else:
            chosen_class = st.selectbox("Flower category", CLASS_LABELS)
            class_index = CLASS_LABELS.index(chosen_class)
            class_paths = [path for path in demo_images if path.parent.name == CLASS_FOLDERS[class_index]]
            if class_paths:
                selected_path = class_paths[len(class_paths) // 2]
                upload_name = selected_path.name
                try:
                    selected_image = Image.open(selected_path).convert("RGB")
                except (OSError, UnidentifiedImageError):
                    st.error("The selected dataset image could not be opened.")

    if selected_image is not None:
        image_hash = hashlib.sha256(selected_image.tobytes()).hexdigest()
        left, right = st.columns([1.05, 1], gap="large")
        with left:
            try:
                st.image(selected_image, caption=upload_name, use_column_width=True)
            except TypeError:
                st.image(selected_image, caption=upload_name, use_container_width=True)
        with right:
            st.markdown("#### Prediction")
            if st.button("Classify image", type="primary", use_container_width=True):
                try:
                    if MODEL_ARTIFACT_PATH.is_file():
                        status_message = "Loading ResNet50 and classifying this image…"
                    else:
                        status_message = (
                            f"Preparing the classifier from up to {MAX_IMAGES_PER_CLASS} images per class. "
                            "This is a one-time setup; features are saved locally."
                        )
                    with st.spinner(status_message):
                        artifacts = load_classifier()
                        probabilities = predict_image(selected_image, artifacts)
                    best_index = int(np.argmax(probabilities))
                    ranked_indices = np.argsort(probabilities)[::-1]
                    st.session_state["last_prediction"] = {
                        "name": CLASS_LABELS[best_index],
                        "probabilities": probabilities.tolist(),
                        "image_hash": image_hash,
                        "filename": upload_name,
                    }
                except Exception as error:
                    st.error(f"Could not run classification: {error}")

            result = st.session_state.get("last_prediction")
            if result and result.get("image_hash") != image_hash:
                result = None
            if result:
                st.metric("Predicted flower", result["name"], f"{max(result['probabilities']) * 100:.1f}% model probability")
                ranked_indices = np.argsort(result["probabilities"])[::-1]
                probability_table = pd.DataFrame(
                    {
                        "Flower": [CLASS_LABELS[index] for index in ranked_indices],
                        "Probability": [result["probabilities"][index] for index in ranked_indices],
                    }
                )
                try:
                    st.bar_chart(probability_table.set_index("Flower"), horizontal=True, color="#286449")
                except TypeError:
                    st.bar_chart(probability_table.set_index("Flower"), color="#286449")
                st.caption("Probabilities are model estimates, not calibrated certainty.")
    else:
        st.info("Upload a flower photo or choose a sample image to begin.")

with evaluation_tab:
    st.subheader("Held-out evaluation")
    st.write("The classifier is evaluated on a stratified 20% holdout from the public five-class flower dataset.")
    if st.button("Load evaluation", key="load_evaluation"):
        try:
            with st.spinner("Loading the pretrained model and evaluating the flower dataset…"):
                artifacts = load_classifier()
            st.session_state["evaluation_loaded"] = True
        except Exception as error:
            st.error(f"Evaluation could not be prepared: {error}")

    if st.session_state.get("evaluation_loaded"):
        artifacts = load_classifier()
        metric_columns = st.columns(3)
        metric_columns[0].metric("Held-out accuracy", f"{artifacts['test_accuracy']:.1%}")
        metric_columns[1].metric("Test images", f"{artifacts['test_count']:,}")
        metric_columns[2].metric("Dataset images", f"{artifacts['image_count']:,}")
        confusion = pd.DataFrame(
            artifacts["confusion_matrix"], index=CLASS_LABELS, columns=CLASS_LABELS
        )
        st.markdown("#### Confusion matrix")
        st.dataframe(confusion, use_container_width=True)
        st.caption("Rows are actual classes; columns are predicted classes.")

with about_tab:
    st.subheader("Workflow")
    st.markdown(
        """
        1. Convert the image to RGB and apply the preprocessing attached to the selected ImageNet weights.
        2. Run inference through frozen ResNet50 and globally pool its feature map into 2,048 values.
        3. Classify those features with a logistic-regression model fitted on the flower dataset.
        4. Display the predicted category and its five class probabilities.

        **No deep network is trained by this app.** Only the small scikit-learn classifier is fitted, and the resulting pipeline is cached for the app session.

        ResNet50 was originally trained to classify ImageNet-1K objects, not flowers. The flower classifier is therefore a transfer-learning application, not a direct use of the original 1,000 ImageNet labels.
        """
    )
    st.markdown("#### Sources and terms")
    st.markdown(
        "- [Torchvision ResNet50 documentation](https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html)\n"
        "- [Torchvision source repository](https://github.com/pytorch/vision)\n"
        "- [TensorFlow flower_photos archive](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz)\n"
        "- Sample images originate from Flickr; verify original creator terms before redistribution."
    )
