"""
FastAPI Backend for Vietnamese Food Classification
Uses the V10 Balanced Final SVM Ensemble model
Image preprocessing identical to training pipeline
"""

import os
import io
import pickle
import logging
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from scipy.stats import skew

# ──────────────────────────────────────────
# Logging
# ──────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────
# App
# ──────────────────────────────────────────
app = FastAPI(
    title="Vietnamese Food Classification API",
    description="Phân loại món ăn Việt Nam sử dụng SVM Ensemble",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────
# Model paths
# ──────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

from huggingface_hub import hf_hub_download
from pathlib import Path
import os

HF_REPO_ID_DEFAULT = "jamus0702/vn_food_classification"

def resolve_model_path() -> Path:
    """
    Ưu tiên local file nếu có (MODELS_DIR/MODEL_FILENAME hoặc MODEL_PATH),
    nếu không có thì tải từ Hugging Face Hub và dùng cache.
    """
    # 1) Nếu bạn muốn chỉ định thẳng path local (dev/prod đều ok)
    env_model_path = os.getenv("MODEL_PATH", "").strip()
    if env_model_path:
        p = Path(env_model_path)
        if p.exists():
            return p

    # 2) Tìm file local theo thư mục models (mặc định ./models)
    models_dir = Path(os.getenv("MODELS_DIR", "models"))
    filename = os.getenv("MODEL_FILENAME", "v10_balanced_cpu.pkl")
    local_candidate = models_dir / filename
    if local_candidate.exists():
        return local_candidate

    # 3) Không có local → tải từ HF (cache)
    # Có thể đổi cache bằng HF_HOME hoặc HF_HUB_CACHE nếu cần. [web:40][web:41]
    hf_repo_id = os.getenv("HF_REPO_ID", HF_REPO_ID_DEFAULT)
    hf_filename = os.getenv("HF_FILENAME", filename)
    hf_token = os.getenv("HF_TOKEN")  # chỉ cần nếu repo private

    downloaded_path = hf_hub_download(
        repo_id=hf_repo_id,
        filename=hf_filename,
        repo_type="model",
        token=hf_token,
    )
    return Path(downloaded_path)

MODEL_PATH = resolve_model_path()


# ──────────────────────────────────────────
# Model state (loaded once at startup)
# ──────────────────────────────────────────
model_state: dict = {}


# ──────────────────────────────────────────
# Feature extraction
# ──────────────────────────────────────────
IMG_SIZE = (256, 256)
USE_SIFT = True


def extract_sift_features(image: np.ndarray) -> np.ndarray:
    """Extract SIFT features (same as training)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=50)
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    if descriptors is None or len(descriptors) == 0:
        return np.zeros(128)

    sift_mean = np.mean(descriptors, axis=0)
    sift_std = np.std(descriptors, axis=0)
    sift_feat = np.concatenate([sift_mean, sift_std])[:128]
    return sift_feat


def calculate_lbp_fast(img: np.ndarray) -> np.ndarray:
    """Manual LBP (same as training)."""
    h, w = img.shape
    lbp = np.zeros((h - 2, w - 2), dtype=np.uint8)
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            center = img[i, j]
            code = 0
            code |= (img[i - 1, j - 1] >= center) << 7
            code |= (img[i - 1, j    ] >= center) << 6
            code |= (img[i - 1, j + 1] >= center) << 5
            code |= (img[i    , j + 1] >= center) << 4
            code |= (img[i + 1, j + 1] >= center) << 3
            code |= (img[i + 1, j    ] >= center) << 2
            code |= (img[i + 1, j - 1] >= center) << 1
            code |= (img[i    , j - 1] >= center) << 0
            lbp[i - 1, j - 1] = code
    return lbp


def extract_features(image: np.ndarray) -> np.ndarray:
    """
    Extract feature vector exactly as in V10_Balanced_Final_CPU.py:
    - RGB histogram (32 bins × 3 channels)
    - HSV histogram (32 bins × 3 channels)
    - LAB histogram (24 bins × 3 channels)
    - HOG features
    - LBP histogram (26 bins)
    - Edge histogram (8 bins)
    - Color moments (mean, std, skew) × 3 channels
    - Color ratios (3 values)
    - SIFT features (128 values)
    """
    features = []

    # RGB histogram
    for i in range(3):
        hist = cv2.calcHist([image], [i], None, [32], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)

    # HSV histogram
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    for i in range(3):
        h_range = [0, 180] if i == 0 else [0, 256]
        hist = cv2.calcHist([hsv], [i], None, [32], h_range)
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)

    # LAB histogram
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    for i in range(3):
        hist = cv2.calcHist([lab], [i], None, [24], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)

    # HOG
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog_img = cv2.resize(gray, (64, 64))
    hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
    hog_features = hog.compute(hog_img)
    features.extend(hog_features.flatten())

    # LBP
    gray_small = cv2.resize(gray, (32, 32))
    lbp = calculate_lbp_fast(gray_small)
    lbp_hist = cv2.calcHist([lbp], [0], None, [26], [0, 256])
    lbp_hist = cv2.normalize(lbp_hist, lbp_hist).flatten()
    features.extend(lbp_hist)

    # Edge histogram
    edges = cv2.Canny(gray, 50, 150)
    edge_hist = cv2.calcHist([edges], [0], None, [8], [0, 256])
    edge_hist = cv2.normalize(edge_hist, edge_hist).flatten()
    features.extend(edge_hist)

    # Color moments
    for i in range(3):
        channel = image[:, :, i].flatten()
        features.append(np.mean(channel) / 255.0)
        features.append(np.std(channel) / 255.0)
        features.append(skew(channel) / 10.0)

    # Color ratios
    b, g, r = cv2.split(image.astype(np.float32) + 1e-6)
    features.append(float(np.mean(r / (b + g))))
    features.append(float(np.mean(g / (r + b))))
    features.append(float(np.mean(b / (r + g))))

    # SIFT
    if USE_SIFT:
        sift_feat = extract_sift_features(image)
        features.extend(sift_feat)

    return np.array(features, dtype=np.float32)


def preprocess_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Decode bytes → BGR image → resize to IMG_SIZE."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Không thể đọc ảnh. Vui lòng gửi file ảnh hợp lệ.")
    img = cv2.resize(img, IMG_SIZE)
    return img


# ──────────────────────────────────────────
# Startup: load model
# ──────────────────────────────────────────
@app.on_event("startup")
async def load_model():
    logger.info(f"Loading model from {MODEL_PATH} ...")
    if not MODEL_PATH.exists():
        logger.warning(f"Model file not found at {MODEL_PATH}")
        return

    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)

    model_state["svm_model"]       = data.get("svm_model")
    model_state["ensemble_models"] = data.get("ensemble_models", [])
    model_state["label_encoder"]   = data["label_encoder"]
    model_state["scaler"]          = data["scaler"]
    model_state["pca"]             = data.get("pca", None)
    model_state["class_names"]     = data["class_names"]
    model_state["img_size"]        = data.get("img_size", (256, 256))
    model_state["use_ensemble"]    = len(model_state["ensemble_models"]) > 0

    logger.info(
        f"Model loaded! Classes: {model_state['class_names']}, "
        f"Ensemble: {model_state['use_ensemble']}"
    )


# ──────────────────────────────────────────
# Response schemas
# ──────────────────────────────────────────
class ClassProbability(BaseModel):
    class_name: str
    probability: float


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    top_classes: List[ClassProbability]
    model_version: str = "V10 Balanced Final CPU (SVM Ensemble)"


# ──────────────────────────────────────────
# Inference helper
# ──────────────────────────────────────────
def run_inference(img: np.ndarray) -> PredictionResponse:
    """Run full inference pipeline on a preprocessed BGR image."""
    if not model_state:
        raise HTTPException(status_code=503, detail="Model chưa được tải. Vui lòng thử lại sau.")

    label_encoder   = model_state["label_encoder"]
    scaler          = model_state["scaler"]
    pca             = model_state["pca"]
    ensemble_models = model_state["ensemble_models"]
    svm_model       = model_state["svm_model"]
    use_ensemble    = model_state["use_ensemble"]

    # Feature extraction
    features = extract_features(img).reshape(1, -1).astype(np.float32)

    # Scale
    features_scaled = scaler.transform(features)

    # PCA
    if pca is not None:
        features_pca = pca.transform(features_scaled.astype(np.float32))
        if hasattr(features_pca, "get"):
            features_pca = features_pca.get()
    else:
        features_pca = features_scaled

    features_pca = features_pca.astype(np.float32)

    # Predict
    if use_ensemble and ensemble_models:
        # Majority vote
        all_preds = []
        all_probs = []
        for model in ensemble_models:
            p = model.predict(features_pca)
            pb = model.predict_proba(features_pca)
            if hasattr(p, "get"):
                p = p.get()
            if hasattr(pb, "get"):
                pb = pb.get()
            all_preds.append(int(p[0]))
            all_probs.append(pb[0])

        from scipy.stats import mode
        prediction = int(mode(all_preds, keepdims=False)[0])
        probabilities = np.mean(all_probs, axis=0)
    else:
        prediction_raw = svm_model.predict(features_pca)
        probs_raw      = svm_model.predict_proba(features_pca)
        if hasattr(prediction_raw, "get"):
            prediction_raw = prediction_raw.get()
        if hasattr(probs_raw, "get"):
            probs_raw = probs_raw.get()
        prediction    = int(prediction_raw[0])
        probabilities = probs_raw[0]

    predicted_class = label_encoder.inverse_transform([prediction])[0]
    confidence = float(probabilities[prediction]) * 100

    # Build top classes (sorted by probability)
    class_names = model_state["class_names"]
    top_classes = sorted(
        [
            ClassProbability(class_name=name, probability=round(float(prob) * 100, 2))
            for name, prob in zip(class_names, probabilities)
        ],
        key=lambda x: x.probability,
        reverse=True,
    )

    return PredictionResponse(
        predicted_class=predicted_class,
        confidence=round(confidence, 2),
        top_classes=top_classes,
    )


# ──────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "message": "Vietnamese Food Classification API",
        "version": "1.0.0",
        "model": "V10 Balanced Final CPU (SVM Ensemble)",
        "classes": model_state.get("class_names", []),
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    loaded = bool(model_state)
    return {
        "status": "ok" if loaded else "model_not_loaded",
        "model_loaded": loaded,
        "classes": model_state.get("class_names", []),
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Upload một ảnh món ăn và nhận kết quả phân loại.

    - **file**: File ảnh (jpg, png, webp, ...)
    - Trả về: tên món ăn dự đoán, độ tin cậy, và xác suất từng lớp
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Vui lòng gửi file ảnh hợp lệ (jpg, png, webp, ...).")

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="File ảnh rỗng.")

    try:
        img = preprocess_image_bytes(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Lỗi khi đọc ảnh")
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý ảnh: {str(e)}")

    try:
        result = run_inference(img)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Lỗi khi chạy inference")
        raise HTTPException(status_code=500, detail=f"Lỗi dự đoán: {str(e)}")

    return result


@app.get("/classes")
async def get_classes():
    """Trả về danh sách các lớp món ăn được hỗ trợ."""
    classes = model_state.get("class_names", [])
    return {"classes": classes, "total": len(classes)}
