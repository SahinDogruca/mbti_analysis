# xgboost_train.py
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import joblib
import warnings
from scipy.stats import uniform, randint
import os
import sys
from sklearn.metrics import confusion_matrix

# Add the parent directory to the path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import (
    EMBEDDINGS_FILE_PATH,
    XGBOOST_MODEL_PATH,
    TEST_SIZE,
    VAL_SIZE,
    RANDOM_STATE,
    USE_RANDOM_SEARCH,
)

warnings.filterwarnings("ignore")


def check_gpu_availability():
    try:
        temp_model = XGBClassifier(
            tree_method="hist",
            predictor="gpu_predictor",
            n_estimators=1,
            enable_categorical=False,
        )
        temp_model.fit(np.array([[0.0, 1.0]], dtype=np.float32), np.array([0]))
        print("XGBoost GPU (CUDA) desteği kullanılabilir.")
        return True
    except Exception as e:
        print(f"XGBoost GPU (CUDA) desteği bulunamadı veya etkinleştirilemedi: {e}")
        return False


USE_GPU = check_gpu_availability()


class MBTIXGBoostAnalyzer:
    def __init__(self, use_random_search=True):
        self.model = None
        self.scaler = None
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.mbti_types = []
        self.use_gpu = USE_GPU
        self.use_random_search = use_random_search

        if self.use_gpu:
            print(f"Model GPU üzerinde eğitilecek.")
        else:
            print(f"Model CPU üzerinde eğitilecek.")

        if self.use_random_search:
            print("Model eğitimi için RandomizedSearchCV kullanılacak.")
        else:
            print("Model eğitimi için sabit XGBoost parametreleri kullanılacak.")

    def load_data(self, filepath):
        print("Veri yükleniyor...")
        data = np.load(filepath, allow_pickle=True)
        features = data["features"]
        labels = data["labels"]
        feature_names = data["feature_names"].tolist()
        features = features.astype(np.float32)
        print(f"Veri yüklendi: {features.shape[0]} sample, {features.shape[1]} feature")
        print(f"MBTI dağılımı: {Counter(labels)}")
        return features, labels, feature_names

    def split_data(
        self, features, labels, test_size=0.2, val_size=0.2, random_state=42
    ):
        print("Veri train/val/test olarak ayrılıyor...")
        encoded_labels = self.label_encoder.fit_transform(labels)
        self.mbti_types = self.label_encoder.classes_
        print(f"MBTI tipleri kodlandı. Sınıflar: {self.mbti_types}")

        X_temp, X_test, y_temp_encoded, y_test_encoded = train_test_split(
            features,
            encoded_labels,
            test_size=test_size,
            random_state=random_state,
            stratify=encoded_labels,
        )

        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train_encoded, y_val_encoded = train_test_split(
            X_temp,
            y_temp_encoded,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=y_temp_encoded,
        )

        print(f"Train set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")

        return X_train, X_val, X_test, y_train_encoded, y_val_encoded, y_test_encoded

    def scale_features(self, X_train, X_val, X_test):
        print("Features normalize ediliyor...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        X_train_scaled = X_train_scaled.astype(np.float32)
        X_val_scaled = X_val_scaled.astype(np.float32)
        X_test_scaled = X_test_scaled.astype(np.float32)
        self.scaler = scaler
        return X_train_scaled, X_val_scaled, X_test_scaled

    def randomized_search_xgboost(self, X_train, y_train_encoded, X_val, y_val_encoded):
        print(f"\n16 MBTI tipi için RandomizedSearchCV başlatılıyor...")
        param_distributions = {
            "n_estimators": randint(100, 500),
            "max_depth": randint(3, 10),
            "learning_rate": uniform(0.01, 0.2),
            "subsample": uniform(0.7, 0.3),
            "colsample_bytree": uniform(0.7, 0.3),
        }
        if self.use_gpu:
            xgb_model = XGBClassifier(
                random_state=RANDOM_STATE,
                eval_metric="mlogloss",
                use_label_encoder=False,
                tree_method="hist",
                predictor="gpu_predictor",
                gpu_id=0,
                num_class=len(self.mbti_types),
                objective="multi:softmax",
            )
            print(f"XGBClassifier GPU ile başlatıldı: 16-tip sınıflandırma")
        else:
            xgb_model = XGBClassifier(
                random_state=RANDOM_STATE,
                eval_metric="mlogloss",
                use_label_encoder=False,
                tree_method="hist",
                num_class=len(self.mbti_types),
                objective="multi:softmax",
            )
            print(f"XGBClassifier CPU ile başlatıldı: 16-tip sınıflandırma")

        random_search = RandomizedSearchCV(
            xgb_model,
            param_distributions,
            n_iter=30,
            cv=2,
            scoring="f1_weighted",
            n_jobs=1 if self.use_gpu else -1,
            verbose=1,
            random_state=RANDOM_STATE,
        )
        random_search.fit(
            X_train, y_train_encoded, eval_set=[(X_val, y_val_encoded)], verbose=False
        )
        print(f"16 MBTI tipi en iyi parametreler: {random_search.best_params_}")
        print(
            f"16 MBTI tipi en iyi CV score (f1_weighted): {random_search.best_score_:.4f}"
        )
        return random_search.best_estimator_

    def train_model(
        self, X_train, X_val, X_test, y_train_encoded, y_val_encoded, y_test_encoded
    ):
        print("\n=== MODEL EĞİTİMİ BAŞLATIYOR ===")
        X_train_scaled, X_val_scaled, X_test_scaled = self.scale_features(
            X_train, X_val, X_test
        )
        if self.use_random_search:
            self.model = self.randomized_search_xgboost(
                X_train_scaled, y_train_encoded, X_val_scaled, y_val_encoded
            )
        else:
            print(
                "RandomizedSearchCV kullanılmadığı için sabit parametrelerle model eğitiliyor."
            )
            if self.use_gpu:
                self.model = XGBClassifier(
                    random_state=RANDOM_STATE,
                    eval_metric="mlogloss",
                    use_label_encoder=False,
                    tree_method="hist",
                    predictor="gpu_predictor",
                    gpu_id=0,
                    num_class=len(self.mbti_types),
                    objective="multi:softmax",
                    n_estimators=379,
                    max_depth=7,
                    learning_rate=0.06426980635477918,
                    subsample=0.807025998008,
                    colsample_bytree=0.8166031869,
                )
            else:
                self.model = XGBClassifier(
                    random_state=RANDOM_STATE,
                    eval_metric="mlogloss",
                    use_label_encoder=False,
                    tree_method="hist",
                    num_class=len(self.mbti_types),
                    objective="multi:softmax",
                    n_estimators=379,
                    max_depth=7,
                    learning_rate=0.06426980635477918,
                    subsample=0.807025998008,
                    colsample_bytree=0.8166031869,
                )
            self.model.fit(
                X_train_scaled,
                y_train_encoded,
                eval_set=[(X_val_scaled, y_val_encoded)],
                verbose=False,
            )
            print("Sabit parametrelerle model eğitimi tamamlandı.")

        val_pred_encoded = self.model.predict(X_val_scaled)
        val_acc = accuracy_score(y_val_encoded, val_pred_encoded)
        print(f"Validation accuracy (16 MBTI tipleri): {val_acc:.4f}")
        return X_train_scaled, X_val_scaled, X_test_scaled

    def predict_mbti(self, X):
        X_scaled = self.scaler.transform(X)
        X_scaled = X_scaled.astype(np.float32)
        predicted_encoded = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        predicted_mbti_types = self.label_encoder.inverse_transform(predicted_encoded)
        confidences = np.array(
            [probabilities[i, pred] for i, pred in enumerate(predicted_encoded)]
        )
        return predicted_mbti_types, confidences

    def evaluate_model(self, X_test, y_test_encoded, dataset_name="Test"):
        print(f"\n=== {dataset_name.upper()} SETİ DEĞERLENDİRMESİ ===")
        predicted_mbti, confidences = self.predict_mbti(X_test)
        y_test_original = self.label_encoder.inverse_transform(y_test_encoded)
        overall_acc = accuracy_score(y_test_original, predicted_mbti)
        print(f"\nOverall MBTI Accuracy: {overall_acc:.4f}")
        print(f"\n--- Detailed Classification Report ---")
        print(
            classification_report(
                y_test_original,
                predicted_mbti,
                target_names=self.mbti_types,
                zero_division=0,
            )
        )
        print(
            "\nBoyut bazlı değerlendirme, tek model senaryosunda doğrudan uygulanmaz."
        )
        return predicted_mbti, confidences, {"overall_accuracy": overall_acc}

    def plot_confusion_matrix(self, y_true_encoded, y_pred, title="Confusion Matrix"):
        y_true_original = self.label_encoder.inverse_transform(y_true_encoded)
        cm = confusion_matrix(y_true_original, y_pred)
        plt.figure(figsize=(16, 14))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.mbti_types,
            yticklabels=self.mbti_types,
        )
        plt.title(title)
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

    def get_feature_importance(self, top_n=20):
        print("\n=== FEATURE IMPORTANCE ANALİZİ ===")
        if not self.feature_names:
            print("Uyarı: feature_names yüklenmedi veya ayarlanmadı.")
            return
        if self.model is None:
            print("Uyarı: Model henüz eğitilmedi.")
            return
        importance = self.model.feature_importances_
        top_indices = np.argsort(importance)[-top_n:][::-1]
        print(f"\n--- Top {top_n} Features (Overall MBTI) ---")
        for i, idx in enumerate(top_indices):
            if idx < len(self.feature_names):
                print(f"{i+1:2d}. {self.feature_names[idx]:<50} {importance[idx]:.4f}")
            else:
                print(f"{i+1:2d}. Unknown Feature (Index: {idx}) {importance[idx]:.4f}")

    def save_model(self, filepath):
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "label_encoder": self.label_encoder,
            "feature_names": self.feature_names,
            "mbti_types": self.mbti_types,
            "use_gpu": self.use_gpu,
            "use_random_search": self.use_random_search,
        }
        joblib.dump(model_data, filepath)
        print(f"Model kaydedildi: {filepath}")

    def load_model(self, filepath):
        model_data = joblib.load(filepath)
        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.label_encoder = model_data["label_encoder"]
        self.feature_names = model_data["feature_names"]
        self.mbti_types = model_data["mbti_types"]
        self.use_gpu = model_data.get("use_gpu", False)
        self.use_random_search = model_data.get("use_random_search", True)
        print(f"Model yüklendi: {filepath}")


def train_main():
    analyzer = MBTIXGBoostAnalyzer(use_random_search=USE_RANDOM_SEARCH)
    features, labels, feature_names = analyzer.load_data(f"{EMBEDDINGS_FILE_PATH}.npz")
    analyzer.feature_names = feature_names

    X_train, X_val, X_test, y_train_encoded, y_val_encoded, y_test_encoded = (
        analyzer.split_data(
            features,
            labels,
            test_size=TEST_SIZE,
            val_size=VAL_SIZE,
            random_state=RANDOM_STATE,
        )
    )

    X_train_scaled, X_val_scaled, X_test_scaled = analyzer.train_model(
        X_train, X_val, X_test, y_train_encoded, y_val_encoded, y_test_encoded
    )

    val_pred, val_conf, val_results = analyzer.evaluate_model(
        X_val, y_val_encoded, "Validation"
    )

    test_pred, test_conf, test_results = analyzer.evaluate_model(
        X_test, y_test_encoded, "Test"
    )

    # Analyzer'ın mbti_types ve label_encoder'ı plot fonksiyonları için güncel olduğundan emin ol
    analyzer.plot_confusion_matrix(
        y_test_encoded, test_pred, "Test Set Confusion Matrix (16 MBTI Types)"
    )
    analyzer.get_feature_importance(top_n=20)
    analyzer.save_model(XGBOOST_MODEL_PATH)


if __name__ == "__main__":
    train_main()
