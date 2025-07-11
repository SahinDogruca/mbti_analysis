# xgboost_train.py
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import joblib
import warnings
from scipy.stats import uniform, randint
import os
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List
from tqdm.auto import tqdm
from pathlib import Path

from config import Config

warnings.filterwarnings("ignore")


# GPU kullanılabilirliğini kontrol et
def check_gpu_availability():
    try:
        temp_model = XGBClassifier(
            tree_method="hist",
            predictor="gpu_predictor",
            n_estimators=1,
            enable_categorical=False,
            objective="multi:softmax",
            num_class=2,
            base_score=0.5,
        )
        temp_model.fit(
            np.array([[0.0, 1.0]], dtype=np.float32),
            np.array([0]),
            eval_set=[(np.array([[0.0, 1.0]], dtype=np.float32), np.array([0]))],
            verbose=False,
        )
        print("XGBoost GPU (CUDA) desteği kullanılabilir.")
        return True
    except Exception as e:
        print(f"XGBoost GPU (CUDA) desteği bulunamadı veya etkinleştirilemedi: {e}")
        return False


# GPU kontrolünü yalnızca bir kez yap
USE_GPU = check_gpu_availability()


class MBTIXGBoostAnalyzer:
    def __init__(self, config_instance: Config):
        self.config = config_instance
        self.model: Optional[XGBClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.label_encoder = LabelEncoder()
        self.feature_names: List[str] = []
        self.mbti_types: List[str] = []

        self.use_gpu = USE_GPU

        if self.use_gpu:
            print(f"Model GPU üzerinde eğitilecek.")
        else:
            print(f"Model CPU üzerinde eğitilecek.")

    def load_data(
        self,
        data_type: str = "mbti_features",  # Default to the single feature type
        version: Optional[str] = None,
    ):
        """NPZ dosyasından veriyi yükle"""
        print(f"Veri yükleniyor: {data_type}...")

        filepath = self.config.get_latest_processed_data_path(
            data_type, extension=".npz"
        )
        if filepath is None:
            raise FileNotFoundError(
                f"En son '{data_type}' özellik dosyası bulunamadı. Lütfen önce `get_embeddings.py`'yi çalıştırın."
            )

        data = np.load(filepath, allow_pickle=True)
        features = data["features"]
        labels = data["labels"]
        feature_names = data["feature_names"].tolist()

        features = features.astype(np.float32)

        print(f"Veri yüklendi: {features.shape[0]} sample, {features.shape[1]} feature")
        print(f"MBTI dağılımı: {Counter(labels)}")

        return features, labels, feature_names

    def split_data(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        test_size: float = 0.2,
        val_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Veriyi train/val/test olarak üçe ayır"""
        print("Veri train/val/test olarak ayrılıyor...")

        encoded_labels = self.label_encoder.fit_transform(labels)
        self.mbti_types = self.label_encoder.classes_.tolist()
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

    def scale_features(
        self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Features'ları normalize et"""
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

    def _get_base_xgb_model(self) -> XGBClassifier:
        """Base XGBoost modelini döndür (GPU/CPU ayarlı)"""
        default_params = self.config.get_model_config("xgboost_multiclass").get(
            "params", {}
        )

        common_params = {
            "random_state": default_params.get("random_state", 42),
            "eval_metric": "mlogloss",
            "num_class": len(self.mbti_types),
            "objective": "multi:softmax",
            "n_estimators": default_params.get("n_estimators", 100),
            "max_depth": default_params.get("max_depth", 6),
            "learning_rate": default_params.get("learning_rate", 0.1),
            "colsample_bytree": default_params.get("colsample_bytree", 0.3),
            "subsample": default_params.get("subsample", 0.3),
        }
        if self.use_gpu:
            return XGBClassifier(
                **common_params, tree_method="hist", predictor="gpu_predictor", gpu_id=0
            )
        else:
            return XGBClassifier(**common_params, tree_method="hist")

    def randomized_search_xgboost(
        self,
        X_train: np.ndarray,
        y_train_encoded: np.ndarray,
        X_val: np.ndarray,
        y_val_encoded: np.ndarray,
    ) -> XGBClassifier:
        """XGBoost için RandomizedSearchCV"""
        print(f"\n16 MBTI tipi için RandomizedSearchCV başlatılıyor...")

        n_iter = self.config.get_data_processing_config().get("tuning_n_iter", 30)
        cv_folds = self.config.get_data_processing_config().get("tuning_cv_folds", 3)

        param_distributions = {
            "n_estimators": randint(100, 500),
            "max_depth": randint(3, 10),
            "learning_rate": uniform(0.01, 0.2),
            "subsample": uniform(0.7, 0.3),
            "colsample_bytree": uniform(0.7, 0.3),
        }

        xgb_model = self._get_base_xgb_model()

        random_search = RandomizedSearchCV(
            xgb_model,
            param_distributions,
            n_iter=n_iter,
            cv=cv_folds,
            scoring="f1_weighted",
            n_jobs=1 if self.use_gpu else -1,
            verbose=1,
            random_state=42,
        )

        random_search.fit(X_train, y_train_encoded)

        print(f"16 MBTI tipi en iyi parametreler: {random_search.best_params_}")
        print(
            f"16 MBTI tipi en iyi CV score (f1_weighted): {random_search.best_score_:.4f}"
        )

        return random_search.best_estimator_

    def train_model(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        y_train_encoded: np.ndarray,
        y_val_encoded: np.ndarray,
        y_test_encoded: np.ndarray,
        fine_tune: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Tüm MBTI tipleri için tek bir XGBoost modeli eğitir.
        fine_tune: Eğer True ise RandomizedSearchCV kullanılır, değilse varsayılan parametrelerle model eğitilir.
        """
        print("\n=== MODEL EĞİTİMİ BAŞLATIYOR ===")

        X_train_scaled, X_val_scaled, X_test_scaled = self.scale_features(
            X_train, X_val, X_test
        )

        if fine_tune:
            self.model = self.randomized_search_xgboost(
                X_train_scaled, y_train_encoded, X_val_scaled, y_val_encoded
            )
        else:
            print(
                "Fine-tuning devre dışı bırakıldı. Varsayılan parametrelerle model eğitiliyor."
            )
            self.model = self._get_base_xgb_model()

            print("Model eğitiliyor (XGBoost verbose çıktısı ile):")
            self.model.fit(
                X_train_scaled,
                y_train_encoded,
                eval_set=[(X_val_scaled, y_val_encoded)],
                verbose=True,
            )

        val_pred_encoded = self.model.predict(X_val_scaled)
        val_acc = accuracy_score(y_val_encoded, val_pred_encoded)

        print(f"Validation accuracy (16 MBTI tipleri): {val_acc:.4f}")

        return X_train_scaled, X_val_scaled, X_test_scaled

    def predict_mbti(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """MBTI tiplerini tahmin et"""
        X_scaled = self.scaler.transform(X) if self.scaler else X
        X_scaled = X_scaled.astype(np.float32)

        predicted_encoded = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)

        predicted_mbti_types = self.label_encoder.inverse_transform(predicted_encoded)

        confidences = np.array(
            [probabilities[i, pred] for i, pred in enumerate(predicted_encoded)]
        )

        return predicted_mbti_types, confidences

    def evaluate_model(
        self, X_test: np.ndarray, y_test_encoded: np.ndarray, dataset_name: str = "Test"
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """Model performansını değerlendir"""
        print(f"\n=== {dataset_name.upper()} SETİ DEĞERLENDİRMESİ ===")

        predicted_mbti, confidences = self.predict_mbti(X_test)

        y_test_original = self.label_encoder.inverse_transform(y_test_encoded)

        overall_acc = accuracy_score(y_test_original, predicted_mbti)
        print(f"\nOverall MBTI Accuracy: {overall_acc:.4f}")

        print("\n--- Detailed Classification Report ---")
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

    def plot_confusion_matrix(
        self,
        y_true_encoded: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Confusion Matrix",
        save_path: Optional[Path] = None,
    ):
        """Confusion matrix çiz"""
        y_true_original = self.label_encoder.inverse_transform(y_true_encoded)

        cm = confusion_matrix(y_true_original, y_pred, labels=self.mbti_types)

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
        if save_path:
            plt.savefig(save_path)
            print(f"Confusion Matrix kaydedildi: {save_path}")
        plt.show()

    def get_feature_importance(self, top_n: int = 20):
        """Feature importance analizi"""
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

    def save_model(self, model_prefix: str = "mbti_xgboost_multiclass_model"):
        """Modeli kaydet"""
        model_filepath = self.config.get_model_path(model_prefix)

        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "label_encoder": self.label_encoder,
            "feature_names": self.feature_names,
            "mbti_types": self.mbti_types,
            "use_gpu": self.use_gpu,
        }

        joblib.dump(model_data, model_filepath)
        print(f"Model kaydedildi: {model_filepath}")

    def load_model(
        self,
        model_prefix: str = "mbti_xgboost_multiclass_model",
        version: Optional[str] = None,
    ):
        """Modeli yükle"""
        if version is None:
            model_filepath = self.config.get_latest_model_path(model_prefix)
            if model_filepath is None:
                raise FileNotFoundError(f"En son '{model_prefix}' modeli bulunamadı.")
        else:
            model_filepath = self.config.get_model_path(model_prefix, version)

        if not model_filepath.exists():
            raise FileNotFoundError(f"Model dosyası bulunamadı: {model_filepath}")

        model_data = joblib.load(model_filepath)

        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.label_encoder = model_data["label_encoder"]
        self.feature_names = model_data["feature_names"]
        self.mbti_types = model_data["mbti_types"]
        self.use_gpu = model_data.get("use_gpu", False)

        print(f"Model yüklendi: {model_filepath}")


# --- Ana Pipeline ---
def run_mbti_analysis(
    config_instance: Config,
    data_type: str = "mbti_features",
    fine_tune_model: bool = True,
):
    """Belirtilen data_type ile MBTI analiz pipeline'ını çalıştırır."""
    print(f"\n{'='*80}")
    print(f"--- MBTI Model Eğitimi ({data_type}) ---")
    print(f"{'='*80}\n")

    analyzer = MBTIXGBoostAnalyzer(config_instance)

    features, labels, feature_names = analyzer.load_data(data_type=data_type)
    analyzer.feature_names = feature_names

    X_train, X_val, X_test, y_train_encoded, y_val_encoded, y_test_encoded = (
        analyzer.split_data(
            features,
            labels,
            test_size=config_instance.get_data_processing_config().get(
                "test_size", 0.2
            ),
            val_size=config_instance.get_data_processing_config().get(
                "validation_size", 0.2
            ),
            random_state=config_instance.get_data_processing_config().get(
                "random_state", 42
            ),
        )
    )

    X_train_scaled, X_val_scaled, X_test_scaled = analyzer.train_model(
        X_train,
        X_val,
        X_test,
        y_train_encoded,
        y_val_encoded,
        y_test_encoded,
        fine_tune=fine_tune_model,
    )

    # Evaluation on Validation Set
    print("\n--- Validation Set Performance ---")
    val_pred, val_conf, val_results = analyzer.evaluate_model(
        X_val_scaled, y_val_encoded, "Validation"
    )
    # Save Confusion Matrix plot for Validation Set
    analyzer.plot_confusion_matrix(
        y_val_encoded,
        val_pred,
        title=f"Validation Set Confusion Matrix ({data_type})",
        save_path=config_instance.get_plots_path(f"confusion_matrix_val_{data_type}"),
    )

    # Evaluation on Test Set
    print("\n--- Test Set Performance ---")
    test_pred, test_conf, test_results = analyzer.evaluate_model(
        X_test_scaled, y_test_encoded, "Test"
    )
    # Save Confusion Matrix plot for Test Set
    analyzer.plot_confusion_matrix(
        y_test_encoded,
        test_pred,
        title=f"Test Set Confusion Matrix ({data_type})",
        save_path=config_instance.get_plots_path(f"confusion_matrix_test_{data_type}"),
    )

    analyzer.get_feature_importance(top_n=20)

    # Save model with a unique prefix based on data_type
    model_save_prefix = f"mbti_xgboost_multiclass_model_{data_type}"
    analyzer.save_model(model_prefix=model_save_prefix)

    print(f"\n=== ÖRNEK PREDİCTİON ({data_type}) ===")
    sample_indices = np.random.choice(len(X_test_scaled), 5, replace=False)
    sample_features = X_test_scaled[sample_indices]
    sample_true_encoded = y_test_encoded[sample_indices]
    sample_true_original = analyzer.label_encoder.inverse_transform(sample_true_encoded)

    sample_pred, sample_conf = analyzer.predict_mbti(sample_features)

    for i in range(5):
        print(
            f"Örnek {i+1}: True={sample_true_original[i]}, Pred={sample_pred[i]}, "
            f"Confidence={sample_conf[i]:.3f}"
        )

    print(f"\nAnaliz tamamlandı for {data_type}! Model ve sonuçlar kaydedildi.")
    return analyzer


if __name__ == "__main__":
    current_config = Config()

    # We now process and use a single type of feature file: 'mbti_features'
    run_mbti_analysis(current_config, data_type="mbti_features", fine_tune_model=False)

    print("\n" + "=" * 80)
    print("Tüm XGBoost eğitim ve değerlendirme işlemleri tamamlandı.")
    print("=" * 80 + "\n")

    # --- Example: Loading and checking the model ---
    print("\n--- Kaydedilen Modeli Yükleme Örneği ---")
    try:
        print("\n'mbti_features' modeli yükleniyor...")
        loaded_analyzer = MBTIXGBoostAnalyzer(current_config)
        loaded_analyzer.load_model(
            model_prefix="mbti_xgboost_multiclass_model_mbti_features"
        )
        print("Model (mbti_features) başarıyla yüklendi.")
        print(
            f"Yüklenen modelin eğitiminde kullanılan özellik sayısı: {len(loaded_analyzer.feature_names)}"
        )

    except FileNotFoundError as e:
        print(f"Model yüklenirken hata oluştu: {e}")
    except Exception as e:
        print(f"Beklenmeyen bir hata oluştu: {e}")
