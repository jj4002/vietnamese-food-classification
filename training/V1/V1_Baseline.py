"""
V1 - BASELINE: SVM RBF with SMOTE
Single SVM model with basic data balancing
"""

import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import skew, kurtosis
import warnings
import gc
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

try:
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTETomek
    IMBLEARN_AVAILABLE = True
    print("SMOTE enabled")
except ImportError:
    IMBLEARN_AVAILABLE = False
    print("SMOTE not available")


class FoodClassifierSVM_Baseline:
    def __init__(self, dataset_path, img_size=(256, 256), use_augmentation=True, 
                 use_pca=True, max_images_per_class=None, use_sift=True, pca_components=350,
                 use_gpu=True, balance_method='smote'):
        
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.max_images_per_class = max_images_per_class
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.pca_components = pca_components
        self.use_augmentation = use_augmentation
        self.use_pca = use_pca
        self.use_sift = use_sift
        self.svm_model = None
        self.class_names = []
        self.best_params = {'C': 5, 'gamma': 'scale', 'kernel': 'rbf'}
        self.balance_method = balance_method
        self.use_gpu = use_gpu and CUML_AVAILABLE
        self.pca = None
        
    def extract_sift_features(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create(nfeatures=100)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        
        if descriptors is None or len(descriptors) == 0:
            return np.zeros(256)
        
        sift_mean = np.mean(descriptors, axis=0)
        sift_std = np.std(descriptors, axis=0)
        sift_feat = np.concatenate([sift_mean, sift_std])
        
        return sift_feat
    
    def extract_features(self, image):
        features = []
        
        for i in range(3):
            hist = cv2.calcHist([image], [i], None, [48], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            features.extend(hist)
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        for i in range(3):
            hist = cv2.calcHist([hsv], [i], None, [48], [0, 256] if i > 0 else [0, 180])
            hist = cv2.normalize(hist, hist).flatten()
            features.extend(hist)
        
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        for i in range(3):
            hist = cv2.calcHist([lab], [i], None, [32], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            features.extend(hist)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hog_img = cv2.resize(gray, (128, 128))
        hog = cv2.HOGDescriptor((128, 128), (16, 16), (8, 8), (8, 8), 9)
        hog_features = hog.compute(hog_img)
        features.extend(hog_features.flatten())
        
        def calculate_lbp(img, points=8, radius=1):
            lbp = np.zeros_like(img)
            for i in range(radius, img.shape[0] - radius):
                for j in range(radius, img.shape[1] - radius):
                    center = img[i, j]
                    binary_string = ''
                    for p in range(points):
                        angle = 2 * np.pi * p / points
                        x = i + int(radius * np.cos(angle))
                        y = j + int(radius * np.sin(angle))
                        if 0 <= x < img.shape[0] and 0 <= y < img.shape[1]:
                            binary_string += '1' if img[x, y] >= center else '0'
                    lbp[i, j] = int(binary_string, 2) if binary_string else 0
            return lbp
        
        gray_small = cv2.resize(gray, (64, 64))
        lbp = calculate_lbp(gray_small)
        lbp_hist = cv2.calcHist([lbp.astype(np.uint8)], [0], None, [32], [0, 256])
        lbp_hist = cv2.normalize(lbp_hist, lbp_hist).flatten()
        features.extend(lbp_hist)
        
        edges = cv2.Canny(gray, 50, 150)
        edge_hist = cv2.calcHist([edges], [0], None, [16], [0, 256])
        edge_hist = cv2.normalize(edge_hist, edge_hist).flatten()
        features.extend(edge_hist)
        
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(sobelx**2 + sobely**2)
        sobel_hist = cv2.calcHist([np.uint8(sobel_mag)], [0], None, [16], [0, 256])
        sobel_hist = cv2.normalize(sobel_hist, sobel_hist).flatten()
        features.extend(sobel_hist)
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_hist = cv2.calcHist([np.uint8(np.abs(laplacian))], [0], None, [32], [0, 256])
        texture_hist = cv2.normalize(texture_hist, texture_hist).flatten()
        features.extend(texture_hist)
        
        for i in range(3):
            channel = image[:, :, i].flatten()
            features.append(np.mean(channel) / 255.0)
            features.append(np.std(channel) / 255.0)
            features.append(skew(channel) / 10.0)
            features.append(kurtosis(channel) / 10.0)
        
        for ksize in [3, 5]:
            sobelx_multi = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
            sobely_multi = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
            sobel_mag_multi = np.sqrt(sobelx_multi**2 + sobely_multi**2)
            features.append(np.mean(sobel_mag_multi) / 255.0)
            features.append(np.std(sobel_mag_multi) / 255.0)
        
        b, g, r = cv2.split(image.astype(np.float32) + 1e-6)
        features.append(np.mean(r / (b + g)))
        features.append(np.mean(g / (r + b)))
        features.append(np.mean(b / (r + g)))
        
        if self.use_sift:
            sift_feat = self.extract_sift_features(image)
            features.extend(sift_feat)
        
        return np.array(features, dtype=np.float32)
    
    def augment_image(self, image, num_augments=5):
        augmented_images = [image]
        
        if num_augments >= 2:
            augmented_images.append(cv2.flip(image, 1))
        
        if num_augments >= 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv_bright = hsv.copy()
            hsv_bright[:, :, 2] = np.clip(hsv_bright[:, :, 2] * 1.15, 0, 255)
            augmented_images.append(cv2.cvtColor(hsv_bright.astype(np.uint8), cv2.COLOR_HSV2BGR))
        
        if num_augments >= 4:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv_dark = hsv.copy()
            hsv_dark[:, :, 2] = np.clip(hsv_dark[:, :, 2] * 0.85, 0, 255)
            augmented_images.append(cv2.cvtColor(hsv_dark.astype(np.uint8), cv2.COLOR_HSV2BGR))
        
        if num_augments >= 5:
            center = (image.shape[1] // 2, image.shape[0] // 2)
            M = cv2.getRotationMatrix2D(center, 8, 1.0)
            augmented_images.append(cv2.warpAffine(image, M, (image.shape[1], image.shape[0])))
        
        return augmented_images[:num_augments]
    
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
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    img = cv2.resize(img, self.img_size)
                    images.append(img)
                    labels.append(class_name)
                except Exception as e:
                    continue
            gc.collect()
        
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        print(f"Imbalance ratio: {max_count/min_count:.2f}x")
        
        return images, labels, class_counts
    
    def balance_data(self, X, y, method='smote'):
        if not IMBLEARN_AVAILABLE:
            return X, y
        
        print(f"Balancing data with {method}...")
        
        if method == 'smote':
            sampler = SMOTE(random_state=42, k_neighbors=5)
        elif method == 'adasyn':
            sampler = ADASYN(random_state=42)
        elif method == 'undersample':
            sampler = RandomUnderSampler(random_state=42)
        elif method == 'combined':
            sampler = SMOTETomek(random_state=42)
        else:
            return X, y
        
        try:
            X_balanced, y_balanced = sampler.fit_resample(X, y)
            print(f"Balanced: {len(X)} -> {len(X_balanced)} samples")
            return X_balanced, y_balanced
        except Exception as e:
            print(f"Balance error: {e}")
            return X, y
    
    def train_with_cv(self, X_train, y_train, X_val, y_val):
        print("Training SVM...")
        print(f"C={self.best_params['C']}, gamma={self.best_params['gamma']}")
        
        if self.best_params['gamma'] == 'scale':
            gamma_val = 1.0 / (X_train.shape[1] * X_train.var())
        elif self.best_params['gamma'] == 'auto':
            gamma_val = 1.0 / X_train.shape[1]
        else:
            gamma_val = self.best_params['gamma']
        
        if self.use_gpu:
            print("Training on GPU...")
            X_train_gpu = X_train.astype(np.float32)
            y_train_gpu = y_train.astype(np.float32)
            
            self.svm_model = cumlSVC(
                C=self.best_params['C'],
                gamma=gamma_val,
                kernel='rbf',
                probability=True,
                cache_size=2000
            )
            self.svm_model.fit(X_train_gpu, y_train_gpu)
        else:
            print("Training on CPU...")
            from sklearn.svm import SVC as sklearnSVC
            self.svm_model = sklearnSVC(
                C=self.best_params['C'],
                gamma=gamma_val,
                kernel='rbf',
                probability=True,
                cache_size=3000,
                class_weight='balanced'
            )
            self.svm_model.fit(X_train, y_train)
        
        X_val_float = X_val.astype(np.float32) if self.use_gpu else X_val
        y_val_pred = self.svm_model.predict(X_val_float)
        if hasattr(y_val_pred, 'get'):
            y_val_pred = y_val_pred.get()
        
        val_acc = accuracy_score(y_val, y_val_pred)
        print(f"Validation Accuracy: {val_acc*100:.2f}%")
        
        return self.svm_model
    
    def train(self, val_size=0.15, test_size=0.15, random_state=42):
        print("="*60)
        print("V1 BASELINE - SVM + SMOTE")
        print("="*60)
        
        images, labels, class_counts = self.load_images()
        labels_encoded = self.label_encoder.fit_transform(labels)
        
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
        
        max_class_count = max(class_counts.values())
        augment_multipliers = {}
        for cls, count in class_counts.items():
            multiplier = min(7, max(3, int(max_class_count / count) + 2))
            augment_multipliers[cls] = multiplier
        
        X_train = []
        y_train_aug = []
        label_to_class = {i: cls for i, cls in enumerate(self.class_names)}
        
        for img, label in tqdm(zip(imgs_train, y_train), total=len(imgs_train), desc="Extracting"):
            class_name = label_to_class[label]
            num_aug = augment_multipliers.get(class_name, 5)
            
            augmented_imgs = self.augment_image(img, num_aug)
            for aug_img in augmented_imgs:
                features = self.extract_features(aug_img)
                X_train.append(features)
                y_train_aug.append(label)
        
        X_train = np.array(X_train, dtype=np.float32)
        y_train = np.array(y_train_aug)
        
        del imgs_train, y_train_aug
        gc.collect()
        
        X_val = np.array([self.extract_features(img) for img in tqdm(imgs_val, desc="Val")], dtype=np.float32)
        X_test = np.array([self.extract_features(img) for img in tqdm(imgs_test, desc="Test")], dtype=np.float32)
        
        del imgs_val, imgs_test
        gc.collect()
        
        print("Scaling...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        del X_train, X_val, X_test
        gc.collect()
        
        print(f"PCA ({self.pca_components} components)...")
        if self.use_gpu:
            self.pca = cumlPCA(n_components=self.pca_components)
            X_train_pca = self.pca.fit_transform(X_train_scaled.astype(np.float32))
            X_val_pca = self.pca.transform(X_val_scaled.astype(np.float32))
            X_test_pca = self.pca.transform(X_test_scaled.astype(np.float32))
            
            if hasattr(X_train_pca, 'get'):
                X_train_pca = X_train_pca.get()
                X_val_pca = X_val_pca.get()
                X_test_pca = X_test_pca.get()
        else:
            self.pca = PCA(n_components=self.pca_components)
            X_train_pca = self.pca.fit_transform(X_train_scaled)
            X_val_pca = self.pca.transform(X_val_scaled)
            X_test_pca = self.pca.transform(X_test_scaled)
        
        del X_train_scaled, X_val_scaled, X_test_scaled
        gc.collect()
        
        if self.balance_method and IMBLEARN_AVAILABLE:
            X_train_balanced, y_train_balanced = self.balance_data(
                X_train_pca, y_train, method=self.balance_method
            )
        else:
            X_train_balanced, y_train_balanced = X_train_pca, y_train
        
        self.train_with_cv(X_train_balanced, y_train_balanced, X_val_pca, y_val)
        
        self.evaluate(X_train_balanced, X_val_pca, X_test_pca, 
                     y_train_balanced, y_val, y_test)
        
        return X_train_balanced, X_val_pca, X_test_pca, y_train_balanced, y_val, y_test
    
    def evaluate(self, X_train, X_val, X_test, y_train, y_val, y_test):
        print("\nEvaluating...")
        
        X_train = X_train.astype(np.float32)
        X_val = X_val.astype(np.float32)
        X_test = X_test.astype(np.float32)
        
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
        plt.title(f'V1 Baseline - Confusion Matrix ({set_name})')
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.tight_layout()
        filename = f'v1_baseline_cm_{set_name}.png'
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
        
        x = np.arange(len(metrics))
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
        ax.set_title(f'V1 Baseline - Classification Report ({set_name})')
        ax.legend(metrics, loc='lower right')
        
        plt.tight_layout()
        filename = f'v1_baseline_report_{set_name}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved {filename}")
        plt.close()
    
    def save_model(self, filename='v1_baseline.pkl'):
        model_data = {
            'svm_model': self.svm_model,
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'pca': self.pca,
            'class_names': self.class_names,
            'img_size': self.img_size,
            'best_params': self.best_params,
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Saved {filename}")
    
    def load_model(self, filename='v1_baseline.pkl'):
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        self.svm_model = model_data['svm_model']
        self.label_encoder = model_data['label_encoder']
        self.scaler = model_data['scaler']
        self.pca = model_data.get('pca', None)
        self.class_names = model_data['class_names']
        self.img_size = model_data['img_size']
        self.best_params = model_data.get('best_params', {})
        
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
        
        prediction = self.svm_model.predict(features_pca.astype(np.float32))
        if hasattr(prediction, 'get'):
            prediction = prediction.get()
        prediction = int(prediction[0])
        
        probabilities = self.svm_model.predict_proba(features_pca.astype(np.float32))
        if hasattr(probabilities, 'get'):
            probabilities = probabilities.get()
        probabilities = probabilities[0]
        
        predicted_class = self.label_encoder.inverse_transform([prediction])[0]
        confidence = probabilities[prediction] * 100
        
        print(f"Prediction: {predicted_class} ({confidence:.1f}%)")
        return predicted_class, confidence


def main():
    print("="*60)
    print("V1 BASELINE - SVM RBF + SMOTE")
    print("="*60)
    
    dataset_path = os.path.join('dataset', 'Train')
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        return
    
    classes = sorted([d for d in os.listdir(dataset_path) 
                     if os.path.isdir(os.path.join(dataset_path, d))])
    
    print(f"{len(classes)} classes found")
    
    classifier = FoodClassifierSVM_Baseline(
        dataset_path,
        img_size=(256, 256),
        use_augmentation=True,
        use_pca=True,
        max_images_per_class=None,
        use_sift=True,
        pca_components=350,
        use_gpu=CUML_AVAILABLE,
        balance_method='smote'
    )
    
    classifier.train(val_size=0.15, test_size=0.15, random_state=42)
    classifier.save_model('v1_baseline.pkl')
    print("Done")


if __name__ == "__main__":
    main()
