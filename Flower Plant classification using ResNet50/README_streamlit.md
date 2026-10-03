# Flower Field Streamlit App

A small image-classification app using pretrained Torchvision ResNet50 features and a flower-specific logistic-regression classifier. ResNet50 remains frozen; the app does not train a deep neural network.

## Run locally

From this folder, install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

Then start the app:

```powershell
python -m streamlit run app.py
```

Open the local URL printed in the terminal (normally `http://localhost:8501`).

## Deploy on Streamlit Community Cloud

The app entry point is `app.py`. Create a GitHub repository for this project and push it from this folder:

```powershell
git init
git add .
git commit -m "Prepare flower classifier for deployment"
git branch -M main
git remote add origin https://github.com/<your-name>/<your-repository>.git
git push -u origin main
```

Then sign in to [Streamlit Community Cloud](https://share.streamlit.io/), create an app from that repository, select the `main` branch, and set the app file to `app.py`.

The repository-root `requirements.txt` installs the same dependencies for Streamlit Community Cloud, whose configured entry point is the root launcher.

The `.gitignore` rules keep the local image dataset and generated feature cache out of Git, while including the small `.cache/flower_classifier_resnet50.joblib` export when it exists. That export avoids fitting the fallback classifier during deployment. The app downloads ResNet50 weights on first classification; dataset examples can be downloaded from inside the app. If the classifier export is absent, first use also downloads the dataset and extracts features on CPU, which can take several minutes.

## Fast predictions and first start

For the fastest prediction path, run the notebook top-to-bottom once, including its final **Export the fitted flower classifier** cell. This saves `.cache/flower_classifier_resnet50.joblib`; Streamlit loads it directly and runs ResNet50 only on the image you select. The notebook's fitted classifier, labels, and evaluation results are reused without retraining.

On first prediction, Torchvision may need to download the pretrained ResNet50 weights. The model is then cached while Streamlit is running. If the notebook-exported classifier is missing, the app falls back to preparing features for up to 250 images per class; that setup can take a few minutes on CPU. The fallback features are saved in `.cache/` and reused after app restarts. The app uses CUDA automatically when available.

## Inputs and outputs

Use **Identify** to upload a JPG, JPEG, PNG, or WebP photograph, or select a sample from the dataset. The result includes one of five labels (daisy, dandelion, rose, sunflower, or tulip) and estimated class probabilities. **Evaluation** displays held-out accuracy and a confusion matrix.

## Model and data sources

- ResNet50 weights and preprocessing: [Torchvision model documentation](https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html)
- Flower dataset: [TensorFlow flower_photos archive](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz)
- Torchvision repository license: BSD-3-Clause. Review current weight terms before redistribution or commercial use.
- Dataset images originate from Flickr; original creator terms still apply. Check them before redistributing images.
