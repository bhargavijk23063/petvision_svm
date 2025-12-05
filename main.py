import os
import glob
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# ------------ CONFIG ------------
CAT_FOLDER = "data/cats/"
DOG_FOLDER = "data/dogs/"
IMAGE_SIZE = (64, 64)     # width, height
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_PCA_COMPONENTS = 100    # reduce dims to speed up SVM (tune if needed)
# --------------------------------

def load_images_from_folder(folder, label, exts=("*.jpg", "*.jpeg", "*.png", "*.bmp")):
    data = []
    labels = []
    for ext in exts:
        for filepath in glob.glob(os.path.join(folder, ext)):
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, IMAGE_SIZE)
            data.append(img)
            labels.append(label)
    return data, labels

def main():
    print("Loading images...")
    cats, cat_labels = load_images_from_folder(CAT_FOLDER, 0)
    dogs, dog_labels = load_images_from_folder(DOG_FOLDER, 1)

    data = cats + dogs
    labels = cat_labels + dog_labels

    if len(data) == 0:
        raise RuntimeError("No images found. Check the folder paths and extensions.")

    data = np.array(data, dtype=np.float32)   # shape: (N, H, W)
    labels = np.array(labels, dtype=np.int32)

    # Flatten images: (N, H*W)
    N, H, W = data.shape
    data = data.reshape(N, H * W)

    # Normalize pixel values to [0,1]
    data /= 255.0

    # Train/test split with stratify to keep label proportions
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels
    )

    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Standardize features (mean=0, var=1) — good before PCA/SVM
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # PCA to reduce dimensionality (speeds up SVM and often improves generalization)
    n_components = min(N_PCA_COMPONENTS, X_train_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    print(f"PCA components: {pca.n_components_}, explained variance ratio (sum): {pca.explained_variance_ratio_.sum():.3f}")

    # Train SVM
    print("Training SVM (this may take a little while)...")
    model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=False, random_state=RANDOM_STATE)
    model.fit(X_train_pca, y_train)

    # Evaluate
    preds = model.predict(X_test_pca)
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.4f}\n")
    print("Classification report:")
    print(classification_report(y_test, preds, target_names=["cat", "dog"]))

    print("Confusion matrix:")
    cm = confusion_matrix(y_test, preds)
    print(cm)

    # Optional: show a few test images with predictions
    try:
        num_show = 8
        indices = np.random.choice(len(X_test), size=min(num_show, len(X_test)), replace=False)
        plt.figure(figsize=(12, 3))
        for i, idx in enumerate(indices):
            img_flat = X_test[idx]
            img = (img_flat.reshape(H, W) * 255).astype(np.uint8)
            pred = preds[idx]
            true = y_test[idx]
            plt.subplot(1, len(indices), i + 1)
            plt.imshow(img, cmap='gray')
            plt.title(f"P:{'dog' if pred==1 else 'cat'}\nT:{'dog' if true==1 else 'cat'}")
            plt.axis('off')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("Could not display sample images (matplotlib/display issue):", e)

if __name__ == "__main__":
    main()

