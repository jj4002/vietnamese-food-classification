"""
V4 - ENSEMBLE: Multiple ML models with Voting
Combines: SVM RBF, SVM Linear, Random Forest, Extra Trees, XGBoost/GB, KNN, Logistic Regression
"""

import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier, 
    ExtraTreesClassifier, 
    GradientBoostingClassifier,
    VotingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import skew
import warnings
import gc
import time
warnings.filterwarnings('ignore')

try:
    from imblearn.over_sampling import SMOTE
    IMBLEARN_AVAILABLE = True
    print("SMOTE enabled")
except ImportError:
    IMBLEARN_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
    print("XGBoost enabled")
except ImportError:
    XGBOOST_AVAILABLE = False


class EnsembleFoodClassifier:
    def __init__(self, dataset_path, img_size=(224, 224), pca_components=300):
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.pca_components = pca_components
        self.pca = None
        self.ensemble_model = None
        self.class_names = []
        self.individual_models = {}
        
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
        hog_img = cv2.resize(gray, (96, 96))
        hog = cv2.HOGDescriptor((96, 96), (16, 16), (8, 8), (8, 8), 9)
        hog_features = hog.compute(hog_img)
        features.extend(hog_features.flatten())
        
        gray_small = cv2.resize(gray, (48, 48))
        gx = cv2.Sobel(gray_small, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_small, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(gx**2 + gy**2)
        texture_hist = cv2.calcHist([np.uint8(mag)], [0], None, [24], [0, 256])
        texture_hist = cv2.normalize(texture_hist, texture_hist).flatten()
        features.extend(texture_hist)
        
        edges = cv2.Canny(gray, 50, 150)
        edge_hist = cv2.calcHist([edges], [0], None, [16], [0, 256])
        edge_hist = cv2.normalize(edge_hist, edge_hist).flatten()
        features.extend(edge_hist)
        
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        lap_hist = cv2.calcHist([np.uint8(np.abs(lap))], [0], None, [20], [0, 256])
        lap_hist = cv2.normalize(lap_hist, lap_hist).flatten()
        features.extend(lap_hist)
        
        for i in range(3):
            channel = image[:, :, i].flatten()
            features.append(np.mean(channel) / 255.0)
            features.append(np.std(channel) / 255.0)
            features.append(skew(channel) / 10.0)
        
        b, g, r = cv2.split(image.astype(np.float32) + 1e-6)
        features.append(np.mean(r / (b + g + 1e-6)))
        features.append(np.mean(g / (r + b + 1e-6)))
        features.append(np.mean(b / (r + g + 1e-6)))
        
        h, w = image.shape[:2]
        quadrants = [
            image[:h//2, :w//2],
            image[:h//2, w//2:],
            image[h//2:, :w//2],
            image[h//2:, w//2:]
        ]
        for quad in quadrants:
            for i in range(3):
                features.append(np.mean(quad[:, :, i]) / 255.0)
        
        return np.array(features, dtype=np.float32)
    
    def augment_image(self, image, num_augments=4):
        augmented = [image]
        
        if num_augments >= 2:
            augmented.append(cv2.flip(image, 1))
        
        if num_augments >= 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.12, 0, 255)
            augmented.append(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR))
        
        if num_augments >= 4:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 0.88, 0, 255)
            augmented.append(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR))
        
        if num_augments >= 5:
            center = (image.shape[1] // 2, image.shape[0] // 2)
            M = cv2.getRotationMatrix2D(center, 5, 1.0)
            augmented.append(cv2.warpAffine(image, M, (image.shape[1], image.shape[0])))
        
        return augmented[:num_augments]
    
    def load_images(self):
        print("Loading images...")
        images, labels = [], []
        
        self.class_names = sorted([d for d in os.listdir(self.dataset_path) 
                                   if os.path.isdir(os.path.join(self.dataset_path, d))])
        
        print(f"Found {len(self.class_names)} classes")
        
        class_counts = {}
        
        for class_name in self.class_names:
            class_path = os.path.join(self.dataset_path, class_name)
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
            
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
                except:
                    continue
            gc.collect()
        
        print(f"Imbalance ratio: {max(class_counts.values())/min(class_counts.values()):.2f}x")
        return images, labels, class_counts
    
    def create_ensemble(self):
        print("\nCreating ensemble models...")
        
        models = []
        
        print("  1. SVM RBF")
        svm_rbf = SVC(C=1.0, kernel='rbf', gamma='scale', probability=True, class_weight='balanced', cache_size=2000)
        models.append(('svm_rbf', svm_rbf))
        
        print("  2. SVM Linear")
        svm_linear = SVC(C=0.5, kernel='linear', probability=True, class_weight='balanced')
        models.append(('svm_linear', svm_linear))
        
        print("  3. Random Forest")
        rf = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=5, min_samples_leaf=2, class_weight='balanced', n_jobs=-1, random_state=42)
        models.append(('random_forest', rf))
        
        print("  4. Extra Trees")
        et = ExtraTreesClassifier(n_estimators=200, max_depth=25, min_samples_split=5, min_samples_leaf=2, class_weight='balanced', n_jobs=-1, random_state=42)
        models.append(('extra_trees', et))
        
        if XGBOOST_AVAILABLE:
            print("  5. XGBoost")
            xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
            models.append(('xgboost', xgb))
        else:
            print("  5. Gradient Boosting")
            gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, subsample=0.8, random_state=42)
            models.append(('gradient_boosting', gb))
        
        print("  6. KNN")
        knn = KNeighborsClassifier(n_neighbors=7, weights='distance', n_jobs=-1)
        models.append(('knn', knn))
        
        print("  7. Logistic Regression")
        lr = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced', n_jobs=-1, random_state=42)
        models.append(('logistic_regression', lr))
        
        print(f"Total: {len(models)} models")
        
        self.ensemble_model = VotingClassifier(estimators=models, voting='soft', n_jobs=-1)
        
        return models
    
    def train(self, val_size=0.15, test_size=0.15, random_state=42):
        start_time = time.time()
        
        print("="*60)
        print("V4 ENSEMBLE - Multiple ML Models")
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
        
        max_count = max(class_counts.values())
        aug_mult = {}
        for cls, count in class_counts.items():
            ratio = max_count / count
            if ratio >= 2.5:
                aug_mult[cls] = 5
            elif ratio >= 1.5:
                aug_mult[cls] = 4
            else:
                aug_mult[cls] = 3
        
        X_train, y_train_aug = [], []
        label_to_class = {i: cls for i, cls in enumerate(self.class_names)}
        
        for img, label in tqdm(zip(imgs_train, y_train), total=len(imgs_train), desc="Extracting"):
            class_name = label_to_class[label]
            num_aug = aug_mult.get(class_name, 3)
            
            for aug_img in self.augment_image(img, num_aug):
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
        self.pca = PCA(n_components=self.pca_components)
        X_train_pca = self.pca.fit_transform(X_train_scaled)
        X_val_pca = self.pca.transform(X_val_scaled)
        X_test_pca = self.pca.transform(X_test_scaled)
        
        explained_var = np.sum(self.pca.explained_variance_ratio_) * 100
        print(f"Explained variance: {explained_var:.1f}%")
        
        del X_train_scaled, X_val_scaled, X_test_scaled
        gc.collect()
        
        if IMBLEARN_AVAILABLE:
            print("SMOTE balancing...")
            smote = SMOTE(random_state=42, k_neighbors=5)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train_pca, y_train)
            print(f"Balanced: {len(X_train_pca)} -> {len(X_train_balanced)}")
        else:
            X_train_balanced, y_train_balanced = X_train_pca, y_train
        
        models = self.create_ensemble()
        
        print("\nTraining individual models...")
        individual_results = []
        
        for name, model in models:
            print(f"  Training {name}...")
            model_start = time.time()
            
            model_clone = clone(model)
            model_clone.fit(X_train_balanced, y_train_balanced)
            
            y_train_pred = model_clone.predict(X_train_balanced)
            y_val_pred = model_clone.predict(X_val_pca)
            y_test_pred = model_clone.predict(X_test_pca)
            
            train_acc = accuracy_score(y_train_balanced, y_train_pred)
            val_acc = accuracy_score(y_val, y_val_pred)
            test_acc = accuracy_score(y_test, y_test_pred)
            gap = (train_acc - test_acc) * 100
            
            model_time = time.time() - model_start
            
            print(f"    {name}: Train={train_acc*100:.1f}%, Test={test_acc*100:.1f}%, Gap={gap:.1f}% ({model_time:.1f}s)")
            
            individual_results.append({
                'name': name,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'test_acc': test_acc,
                'gap': gap,
                'model': model_clone
            })
            
            self.individual_models[name] = model_clone
        
        print("\nTraining ensemble...")
        ensemble_start = time.time()
        self.ensemble_model.fit(X_train_balanced, y_train_balanced)
        print(f"Ensemble trained in {time.time() - ensemble_start:.1f}s")
        
        print("\nEvaluating...")
        y_train_pred = self.ensemble_model.predict(X_train_balanced)
        y_val_pred = self.ensemble_model.predict(X_val_pca)
        y_test_pred = self.ensemble_model.predict(X_test_pca)
        
        train_acc = accuracy_score(y_train_balanced, y_train_pred)
        val_acc = accuracy_score(y_val, y_val_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        
        train_f1 = f1_score(y_train_balanced, y_train_pred, average='weighted')
        val_f1 = f1_score(y_val, y_val_pred, average='weighted')
        test_f1 = f1_score(y_test, y_test_pred, average='weighted')
        
        gap = (train_acc - test_acc) * 100
        
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        print(f"{'Model':<25} {'Train':>8} {'Val':>8} {'Test':>8} {'Gap':>8}")
        print("-"*60)
        
        for result in individual_results:
            print(f"{result['name']:<25} {result['train_acc']*100:>7.1f}% {result['val_acc']*100:>7.1f}% {result['test_acc']*100:>7.1f}% {result['gap']:>7.1f}%")
        
        print("-"*60)
        print(f"{'ENSEMBLE':<25} {train_acc*100:>7.1f}% {val_acc*100:>7.1f}% {test_acc*100:>7.1f}% {gap:>7.1f}%")
        print("="*60)
        
        best_individual = max(individual_results, key=lambda x: x['test_acc'])
        print(f"\nBest individual: {best_individual['name']} ({best_individual['test_acc']*100:.1f}%)")
        print(f"Ensemble: {test_acc*100:.1f}%")
        
        target_names = self.label_encoder.classes_
        print("\nTest Report:")
        print(classification_report(y_test, y_test_pred, target_names=target_names))
        
        self.plot_confusion_matrix(y_test, y_test_pred, target_names)
        self.plot_classification_report(y_test, y_test_pred, target_names)
        self.plot_model_comparison(individual_results, train_acc, val_acc, test_acc)
        
        print(f"\nTotal time: {(time.time() - start_time)/60:.1f} min")
        
        self.individual_results = individual_results
        
        return X_train_balanced, X_val_pca, X_test_pca, y_train_balanced, y_val, y_test
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names):
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('V4 Ensemble - Confusion Matrix (Test)')
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig('v4_ensemble_cm.png', dpi=300)
        print("Saved v4_ensemble_cm.png")
        plt.close()
    
    def plot_classification_report(self, y_true, y_pred, class_names):
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
        ax.set_title('V4 Ensemble - Classification Report (Test)')
        ax.legend(metrics, loc='lower right')
        
        plt.tight_layout()
        plt.savefig('v4_ensemble_report.png', dpi=300, bbox_inches='tight')
        print("Saved v4_ensemble_report.png")
        plt.close()
    
    def plot_model_comparison(self, individual_results, ensemble_train, ensemble_val, ensemble_test):
        models = [r['name'] for r in individual_results] + ['ENSEMBLE']
        train_accs = [r['train_acc']*100 for r in individual_results] + [ensemble_train*100]
        val_accs = [r['val_acc']*100 for r in individual_results] + [ensemble_val*100]
        test_accs = [r['test_acc']*100 for r in individual_results] + [ensemble_test*100]
        
        x = np.arange(len(models))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        ax.bar(x - width, train_accs, width, label='Train', color='#3498db')
        ax.bar(x, val_accs, width, label='Validation', color='#2ecc71')
        ax.bar(x + width, test_accs, width, label='Test', color='#e74c3c')
        
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('V4 Ensemble - Model Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 105)
        
        plt.tight_layout()
        plt.savefig('v4_model_comparison.png', dpi=300)
        print("Saved v4_model_comparison.png")
        plt.close()
    
    def save_model(self, filename='v4_ensemble.pkl'):
        model_data = {
            'ensemble_model': self.ensemble_model,
            'individual_models': self.individual_models,
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'pca': self.pca,
            'class_names': self.class_names,
            'img_size': self.img_size,
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Saved {filename}")
    
    def load_model(self, filename='v4_ensemble.pkl'):
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        self.ensemble_model = model_data['ensemble_model']
        self.individual_models = model_data.get('individual_models', {})
        self.label_encoder = model_data['label_encoder']
        self.scaler = model_data['scaler']
        self.pca = model_data.get('pca', None)
        self.class_names = model_data['class_names']
        self.img_size = model_data['img_size']
        
        print(f"Loaded {filename}")
    
    def predict(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read: {image_path}")
        
        img = cv2.resize(img, self.img_size)
        features = self.extract_features(img).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        features_pca = self.pca.transform(features_scaled)
        
        prediction = self.ensemble_model.predict(features_pca)[0]
        probabilities = self.ensemble_model.predict_proba(features_pca)[0]
        
        predicted_class = self.label_encoder.inverse_transform([prediction])[0]
        confidence = probabilities[prediction] * 100
        
        print(f"Prediction: {predicted_class} ({confidence:.1f}%)")
        return predicted_class, confidence


def main():
    print("="*60)
    print("V4 ENSEMBLE - Multiple ML Models")
    print("="*60)
    
    dataset_path = os.path.join('dataset', 'Train')
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        return
    
    classes = sorted([d for d in os.listdir(dataset_path) 
                     if os.path.isdir(os.path.join(dataset_path, d))])
    
    print(f"{len(classes)} classes found")
    
    classifier = EnsembleFoodClassifier(
        dataset_path,
        img_size=(224, 224),
        pca_components=300
    )
    
    classifier.train(val_size=0.15, test_size=0.15, random_state=42)
    classifier.save_model('v4_ensemble.pkl')
    print("Done")


if __name__ == "__main__":
    main()
