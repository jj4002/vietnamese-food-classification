"""
V10 - BALANCED FINAL: SVM Ensemble with Anti-Overfitting
Features: Feature noise, Feature dropout, Subsampling, No SMOTE
"""

import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import skew, mode
import warnings
import gc
import time
warnings.filterwarnings('ignore')

try:
    from cuml.svm import SVC as cumlSVC
    from cuml.decomposition import PCA as cumlPCA
    CUML_AVAILABLE = True
    print("GPU detected")
except ImportError:
    from sklearn.svm import SVC as sklearnSVC
    CUML_AVAILABLE = False
    print("Using CPU")


class FoodClassifierSVM_Balanced:
    
    def __init__(self, dataset_path, img_size=(256, 256), 
                 use_augmentation=True, use_pca=True, 
                 max_images_per_class=None, use_sift=True, 
                 pca_variance=0.92, use_gpu=True,
                 train_subsample_ratio=0.85,
                 feature_noise_std=0.05,
                 feature_dropout_rate=0.05,
                 c_value=0.3,
                 use_ensemble=True,
                 n_ensemble=3):
        
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.max_images_per_class = max_images_per_class
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.pca_variance = pca_variance
        self.use_augmentation = use_augmentation
        self.use_pca = use_pca
        self.use_sift = use_sift
        self.svm_model = None
        self.class_names = []
        self.use_gpu = use_gpu and CUML_AVAILABLE
        self.pca = None
        self.train_subsample_ratio = train_subsample_ratio
        self.feature_noise_std = feature_noise_std
        self.feature_dropout_rate = feature_dropout_rate
        self.c_value = c_value
        self.use_ensemble = use_ensemble
        self.n_ensemble = n_ensemble
        self.ensemble_models = []
        self.best_params = {'C': c_value, 'gamma': 'scale', 'kernel': 'rbf'}
        
    def extract_sift_features(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create(nfeatures=50)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        
        if descriptors is None or len(descriptors) == 0:
            return np.zeros(128)
        
        sift_mean = np.mean(descriptors, axis=0)
        sift_std = np.std(descriptors, axis=0)
        sift_feat = np.concatenate([sift_mean, sift_std])[:128]
        return sift_feat
    
    def extract_features(self, image):
        features = []
        
        for i in range(3):
            hist = cv2.calcHist([image], [i], None, [32], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            features.extend(hist)
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        for i in range(3):
            hist = cv2.calcHist([hsv], [i], None, [32], [0, 256] if i > 0 else [0, 180])
            hist = cv2.normalize(hist, hist).flatten()
            features.extend(hist)
        
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        for i in range(3):
            hist = cv2.calcHist([lab], [i], None, [24], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            features.extend(hist)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hog_img = cv2.resize(gray, (64, 64))
        hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
        hog_features = hog.compute(hog_img)
        features.extend(hog_features.flatten())
        
        def calculate_lbp_fast(img):
            h, w = img.shape
            lbp = np.zeros((h-2, w-2), dtype=np.uint8)
            for i in range(1, h-1):
                for j in range(1, w-1):
                    center = img[i, j]
                    code = 0
                    code |= (img[i-1, j-1] >= center) << 7
                    code |= (img[i-1, j  ] >= center) << 6
                    code |= (img[i-1, j+1] >= center) << 5
                    code |= (img[i  , j+1] >= center) << 4
                    code |= (img[i+1, j+1] >= center) << 3
                    code |= (img[i+1, j  ] >= center) << 2
                    code |= (img[i+1, j-1] >= center) << 1
                    code |= (img[i  , j-1] >= center) << 0
                    lbp[i-1, j-1] = code
            return lbp
        
        gray_small = cv2.resize(gray, (32, 32))
        lbp = calculate_lbp_fast(gray_small)
        lbp_hist = cv2.calcHist([lbp], [0], None, [26], [0, 256])
        lbp_hist = cv2.normalize(lbp_hist, lbp_hist).flatten()
        features.extend(lbp_hist)
        
        edges = cv2.Canny(gray, 50, 150)
        edge_hist = cv2.calcHist([edges], [0], None, [8], [0, 256])
        edge_hist = cv2.normalize(edge_hist, edge_hist).flatten()
        features.extend(edge_hist)
        
        for i in range(3):
            channel = image[:, :, i].flatten()
            features.append(np.mean(channel) / 255.0)
            features.append(np.std(channel) / 255.0)
            features.append(skew(channel) / 10.0)
        
        b, g, r = cv2.split(image.astype(np.float32) + 1e-6)
        features.append(np.mean(r / (b + g)))
        features.append(np.mean(g / (r + b)))
        features.append(np.mean(b / (r + g)))
        
        if self.use_sift:
            sift_feat = self.extract_sift_features(image)
            features.extend(sift_feat)
        
        return np.array(features, dtype=np.float32)
    
    def augment_image(self, image, num_augments=4):
        augmented_images = [image]
        
        if num_augments >= 2:
            augmented_images.append(cv2.flip(image, 1))
        
        if num_augments >= 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.15, 0, 255)
            augmented_images.append(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR))
        
        if num_augments >= 4:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 0.85, 0, 255)
            augmented_images.append(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR))
        
        return augmented_images[:num_augments]
    
    def add_feature_noise(self, X, noise_std=0.1, random_state=None):
        if random_state is not None:
            np.random.seed(random_state)
        noise = np.random.normal(0, noise_std, X.shape).astype(np.float32)
        return X + noise
    
    def apply_feature_dropout(self, X, dropout_rate=0.1, random_state=None):
        if random_state is not None:
            np.random.seed(random_state)
        mask = np.random.binomial(1, 1 - dropout_rate, X.shape).astype(np.float32)
        return X * mask
    
    def subsample_training_data(self, X, y, ratio=0.7, random_state=None):
        if random_state is not None:
            np.random.seed(random_state)
        n_samples = int(len(X) * ratio)
        indices = np.random.choice(len(X), n_samples, replace=False)
        return X[indices], y[indices]
    
    def load_images(self):
        print("Loading images...")
        images = []
        labels = []
        
        self.class_names = sorted([d for d in os.listdir(self.dataset_path) 
                                   if os.path.isdir(os.path.join(self.dataset_path, d))])
        
        print(f"Found {len(self.class_names)} classes")
        
        class_counts = {}
        
        for class_name in self.class_names:
            class_path = os.path.join(self.dataset_path, class_name)
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
            
            if self.max_images_per_class and len(image_files) > self.max_images_per_class:
                np.random.seed(42)
                image_files = np.random.choice(image_files, self.max_images_per_class, 
                                              replace=False).tolist()
            
            class_counts[class_name] = len(image_files)
            print(f"  {class_name}: {len(image_files)} images")
            
            for img_file in tqdm(image_files, desc=class_name, leave=False):
                img_path = os.path.join(class_path, img_file)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, self.img_size)
                    images.append(img)
                    labels.append(class_name)
            gc.collect()
        
        return images, labels, class_counts
    
    def train_single_model(self, X_train, y_train, model_idx=0, random_state=42):
        base_seed = random_state + model_idx * 100
        
        X_train_noisy = self.add_feature_noise(X_train.copy(), self.feature_noise_std, random_state=base_seed)
        X_train_dropped = self.apply_feature_dropout(X_train_noisy, self.feature_dropout_rate, random_state=base_seed + 1)
        X_train_sub, y_train_sub = self.subsample_training_data(
            X_train_dropped, y_train, self.train_subsample_ratio, random_state=base_seed + 2
        )
        
        print(f"  Model {model_idx+1}: {len(X_train_sub)} samples")
        
        if self.use_gpu:
            X_train_gpu = X_train_sub.astype(np.float32)
            y_train_gpu = y_train_sub.astype(np.float32)
            gamma_val = 1.0 / (X_train_sub.shape[1] * np.var(X_train_sub))
            model = cumlSVC(C=self.c_value, gamma=gamma_val, kernel='rbf', 
                           probability=True, cache_size=2000)
            model.fit(X_train_gpu, y_train_gpu)
        else:
            from sklearn.svm import SVC
            model = SVC(C=self.c_value, gamma='scale', kernel='rbf',
                       probability=True, cache_size=3000, class_weight='balanced')
            model.fit(X_train_sub, y_train_sub)
        
        return model
    
    def train_ensemble(self, X_train, y_train, random_state=42):
        print(f"Training ensemble ({self.n_ensemble} models)")
        print(f"  C={self.c_value}, subsample={self.train_subsample_ratio}")
        
        self.ensemble_models = []
        for i in range(self.n_ensemble):
            model = self.train_single_model(X_train, y_train, model_idx=i, random_state=random_state)
            self.ensemble_models.append(model)
        
        self.svm_model = self.ensemble_models[0]
        return self.ensemble_models
    
    def ensemble_predict(self, X):
        X = X.astype(np.float32)
        all_predictions = []
        for model in self.ensemble_models:
            preds = model.predict(X)
            if hasattr(preds, 'get'):
                preds = preds.get()
            all_predictions.append(preds)
        
        all_predictions = np.array(all_predictions)
        final_predictions = mode(all_predictions, axis=0, keepdims=False)[0]
        return final_predictions
    
    def ensemble_predict_proba(self, X):
        X = X.astype(np.float32)
        all_probs = []
        for model in self.ensemble_models:
            probs = model.predict_proba(X)
            if hasattr(probs, 'get'):
                probs = probs.get()
            all_probs.append(probs)
        return np.mean(all_probs, axis=0)
    
    def train(self, val_size=0.15, test_size=0.15, random_state=42):
        np.random.seed(random_state)
        
        start_time = time.time()
        
        print("="*60)
        print("V10 BALANCED FINAL - SVM Ensemble + Anti-Overfitting")
        print("="*60)
        
        images, labels, class_counts = self.load_images()
        labels_encoded = self.label_encoder.fit_transform(labels)
        
        print("Splitting data...")
        imgs_train, imgs_temp, y_train, y_temp = train_test_split(
            images, labels_encoded, test_size=(val_size + test_size),
            random_state=random_state, stratify=labels_encoded
        )
        
        imgs_val, imgs_test, y_val, y_test = train_test_split(
            imgs_temp, y_temp, test_size=test_size/(val_size + test_size),
            random_state=random_state, stratify=y_temp
        )
        
        del images, labels, labels_encoded, imgs_temp, y_temp
        gc.collect()
        
        print(f"Train: {len(imgs_train)}, Val: {len(imgs_val)}, Test: {len(imgs_test)}")
        
        print("Extracting features...")
        max_class_count = max(class_counts.values())
        augment_multipliers = {}
        for cls, count in class_counts.items():
            ratio = max_class_count / count
            if ratio >= 2.5:
                multiplier = 5
            elif ratio >= 1.5:
                multiplier = 4
            else:
                multiplier = 3
            augment_multipliers[cls] = multiplier
        
        X_train = []
        y_train_aug = []
        label_to_class = {i: cls for i, cls in enumerate(self.class_names)}
        
        for img, label in tqdm(zip(imgs_train, y_train), total=len(imgs_train)):
            class_name = label_to_class[label]
            num_aug = augment_multipliers.get(class_name, 3)
            augmented_imgs = self.augment_image(img, num_aug)
            for aug_img in augmented_imgs:
                features = self.extract_features(aug_img)
                X_train.append(features)
                y_train_aug.append(label)
        
        X_train = np.array(X_train, dtype=np.float32)
        y_train = np.array(y_train_aug)
        del imgs_train, y_train_aug
        gc.collect()
        
        X_val = np.array([self.extract_features(img) for img in tqdm(imgs_val)], dtype=np.float32)
        X_test = np.array([self.extract_features(img) for img in tqdm(imgs_test)], dtype=np.float32)
        del imgs_val, imgs_test
        gc.collect()
        
        print("Scaling...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        del X_train, X_val, X_test
        gc.collect()
        
        print(f"PCA (variance={self.pca_variance})...")
        if self.use_gpu:
            n_comp = min(400, X_train_scaled.shape[1])
            self.pca = cumlPCA(n_components=n_comp)
            X_train_pca_full = self.pca.fit_transform(X_train_scaled.astype(np.float32))
            
            if hasattr(X_train_pca_full, 'get'):
                X_train_pca_full = X_train_pca_full.get()
            
            explained_var_ratio = self.pca.explained_variance_ratio_
            if hasattr(explained_var_ratio, 'get'):
                explained_var_ratio = explained_var_ratio.get()
            
            cumsum = np.cumsum(explained_var_ratio)
            n_comp_opt = np.searchsorted(cumsum, self.pca_variance) + 1
            n_comp_opt = min(n_comp_opt, n_comp)
            
            self.pca = cumlPCA(n_components=n_comp_opt)
            X_train_pca = self.pca.fit_transform(X_train_scaled.astype(np.float32))
            X_val_pca = self.pca.transform(X_val_scaled.astype(np.float32))
            X_test_pca = self.pca.transform(X_test_scaled.astype(np.float32))
            
            if hasattr(X_train_pca, 'get'):
                X_train_pca = X_train_pca.get()
                X_val_pca = X_val_pca.get()
                X_test_pca = X_test_pca.get()
        else:
            self.pca = PCA(n_components=self.pca_variance)
            X_train_pca = self.pca.fit_transform(X_train_scaled)
            X_val_pca = self.pca.transform(X_val_scaled)
            X_test_pca = self.pca.transform(X_test_scaled)
        
        del X_train_scaled, X_val_scaled, X_test_scaled
        gc.collect()
        
        print(f"PCA done: {X_train_pca.shape[1]} components")
        
        if self.use_ensemble:
            self.train_ensemble(X_train_pca, y_train, random_state=random_state)
        else:
            self.svm_model = self.train_single_model(X_train_pca, y_train, random_state=random_state)
        
        print(f"Training took {(time.time() - start_time)/60:.1f} min")
        
        self.evaluate(X_train_pca, X_val_pca, X_test_pca, y_train, y_val, y_test)
        return X_train_pca, X_val_pca, X_test_pca, y_train, y_val, y_test
    
    def evaluate(self, X_train, X_val, X_test, y_train, y_val, y_test):
        print("\nEvaluating...")
        
        X_train = X_train.astype(np.float32)
        X_val = X_val.astype(np.float32)
        X_test = X_test.astype(np.float32)
        
        if self.use_ensemble and len(self.ensemble_models) > 0:
            y_train_pred = self.ensemble_predict(X_train)
            y_val_pred = self.ensemble_predict(X_val)
            y_test_pred = self.ensemble_predict(X_test)
        else:
            y_train_pred = self.svm_model.predict(X_train)
            y_val_pred = self.svm_model.predict(X_val)
            y_test_pred = self.svm_model.predict(X_test)
            
            if hasattr(y_train_pred, 'get'):
                y_train_pred = y_train_pred.get()
                y_val_pred = y_val_pred.get()
                y_test_pred = y_test_pred.get()
        
        train_acc = accuracy_score(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred, average='weighted')
        val_acc = accuracy_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred, average='weighted')
        test_acc = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred, average='weighted')
        
        print(f"Train Acc: {train_acc*100:.2f}%  |  F1: {train_f1*100:.2f}%")
        print(f"Val Acc: {val_acc*100:.2f}%  |  F1: {val_f1*100:.2f}%")
        print(f"Test Acc: {test_acc*100:.2f}%  |  F1: {test_f1*100:.2f}%")
        print(f"Gap: {(train_acc - test_acc)*100:.2f}%")
        
        target_names = self.label_encoder.classes_
        print("\nTest Report:")
        print(classification_report(y_test, y_test_pred, target_names=target_names))
        
        self.plot_confusion_matrix(y_test, y_test_pred, target_names, 'test')
        self.plot_confusion_matrix(y_val, y_val_pred, target_names, 'val')
        self.plot_classification_report(y_test, y_test_pred, target_names, 'test')
        self.plot_classification_report(y_val, y_val_pred, target_names, 'val')
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names, set_name='test'):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'V10 Balanced - Confusion Matrix ({set_name})')
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.tight_layout()
        filename = f'v10_balanced_cm_{set_name}.png'
        plt.savefig(filename, dpi=300)
        print(f"Saved {filename}")
        plt.close()
    
    def plot_classification_report(self, y_true, y_pred, class_names, set_name='test'):
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        
        metrics = ['precision', 'recall', 'f1-score']
        data = []
        labels = []
        
        for class_name in class_names:
            if class_name in report:
                labels.append(class_name)
                data.append([report[class_name][m] for m in metrics])
        
        labels.extend(['macro avg', 'weighted avg'])
        data.append([report['macro avg'][m] for m in metrics])
        data.append([report['weighted avg'][m] for m in metrics])
        
        data = np.array(data)
        
        fig, ax = plt.subplots(figsize=(10, len(labels) * 0.6 + 2))
        
        height = 0.25
        colors = ['#3498db', '#2ecc71', '#e74c3c']
        
        for i, label in enumerate(labels):
            for j, metric in enumerate(metrics):
                ax.barh(i + (j - 1) * height, data[i, j], height, color=colors[j], alpha=0.8)
                ax.text(data[i, j] + 0.01, i + (j - 1) * height, f'{data[i, j]:.2f}', va='center', fontsize=8)
        
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel('Score')
        ax.set_xlim(0, 1.15)
        ax.set_title(f'V10 Balanced - Classification Report ({set_name})')
        ax.legend(metrics, loc='lower right')
        
        plt.tight_layout()
        filename = f'v10_balanced_report_{set_name}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved {filename}")
        plt.close()
    
    def save_model(self, filename='v10_balanced.pkl'):
        model_data = {
            'svm_model': self.svm_model,
            'ensemble_models': self.ensemble_models if self.use_ensemble else [],
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'pca': self.pca,
            'class_names': self.class_names,
            'img_size': self.img_size,
            'best_params': self.best_params,
            'c_value': self.c_value,
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Saved {filename}")
    
    def load_model(self, filename='v10_balanced.pkl'):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        self.svm_model = data['svm_model']
        self.ensemble_models = data.get('ensemble_models', [])
        self.label_encoder = data['label_encoder']
        self.scaler = data['scaler']
        self.pca = data.get('pca', None)
        self.class_names = data['class_names']
        self.img_size = data['img_size']
        print(f"Loaded {filename}")
    
    def predict(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read: {image_path}")
        
        img = cv2.resize(img, self.img_size)
        features = self.extract_features(img).reshape(1, -1).astype(np.float32)
        features_scaled = self.scaler.transform(features)
        
        if self.pca is not None:
            features_pca = self.pca.transform(features_scaled.astype(np.float32))
            if hasattr(features_pca, 'get'):
                features_pca = features_pca.get()
        else:
            features_pca = features_scaled
        
        if self.use_ensemble and len(self.ensemble_models) > 0:
            prediction = self.ensemble_predict(features_pca.astype(np.float32))
            probabilities = self.ensemble_predict_proba(features_pca.astype(np.float32))[0]
        else:
            prediction = self.svm_model.predict(features_pca.astype(np.float32))
            if hasattr(prediction, 'get'):
                prediction = prediction.get()
            probabilities = self.svm_model.predict_proba(features_pca.astype(np.float32))
            if hasattr(probabilities, 'get'):
                probabilities = probabilities.get()
            probabilities = probabilities[0]
        
        prediction = int(prediction[0]) if hasattr(prediction, '__len__') else int(prediction)
        predicted_class = self.label_encoder.inverse_transform([prediction])[0]
        confidence = probabilities[prediction] * 100
        
        print(f"Prediction: {predicted_class} ({confidence:.1f}%)")
        return predicted_class, confidence


def main():
    print("="*60)
    print("V10 BALANCED FINAL - SVM Ensemble + Anti-Overfitting")
    print("="*60)
    
    dataset_path = os.path.join('dataset', 'Train')
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        return
    
    classes = sorted([d for d in os.listdir(dataset_path) 
                     if os.path.isdir(os.path.join(dataset_path, d))])
    
    print(f"{len(classes)} classes found")
    
    if CUML_AVAILABLE:
        gpu_choice = input("Use GPU or CPU? (gpu/cpu) [gpu]: ").strip().lower() or 'gpu'
        use_gpu = (gpu_choice == 'gpu')
    else:
        use_gpu = False
    
    classifier = FoodClassifierSVM_Balanced(
        dataset_path,
        img_size=(256, 256),
        use_augmentation=True,
        use_pca=True,
        max_images_per_class=None,
        use_sift=True,
        pca_variance=0.92,
        use_gpu=use_gpu,
        train_subsample_ratio=0.85,
        feature_noise_std=0.05,
        feature_dropout_rate=0.05,
        c_value=0.3,
        use_ensemble=True,
        n_ensemble=3
    )
    
    classifier.train(val_size=0.15, test_size=0.15, random_state=42)
    classifier.save_model('v10_balanced.pkl')
    print("Done")


if __name__ == "__main__":
    main()
