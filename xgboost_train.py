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
from imblearn.over_sampling import SMOTE  # SMOTE için gerekli import
from sklearn.utils.class_weight import (
    compute_class_weight,
)  # Sınıf ağırlıkları için gerekli import


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import (
    EMBEDDINGS_FILE_PATH,
    XGBOOST_MODEL_PATH,
    EMBEDDINGS_FILE_PATH_BERT,
    USE_BERT_EMBEDDINGS,
    TEST_SIZE,
    VAL_SIZE,
    RANDOM_STATE,
    USE_RANDOM_SEARCH,
    CLASS_IMBALANCE_STRATEGY,  # Yeni eklendi
    SMOTE_K_NEIGHBORS,  # Yeni eklendi
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
    def __init__(
        self, use_random_search=True, imbalance_strategy="none", smote_k_neighbors=5
    ):  # Parametreler eklendi
        self.model = None
        self.scaler = None
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.mbti_types = []
        self.use_gpu = USE_GPU
        self.use_random_search = use_random_search
        self.imbalance_strategy = imbalance_strategy  # Yeni
        self.smote_k_neighbors = smote_k_neighbors  # Yeni

        if self.use_gpu:
            print("Model GPU üzerinde eğitilecek.")
        else:
            print("Model CPU üzerinde eğitilecek.")

        if self.use_random_search:
            print("Model eğitimi için RandomizedSearchCV kullanılacak.")
        else:
            print("Model eğitimi için sabit XGBoost parametreleri kullanılacak.")

        print(f"Sınıf dengesizliği stratejisi: {self.imbalance_strategy}")
        if self.imbalance_strategy == "smote":
            print(f"SMOTE k_neighbors: {self.smote_k_neighbors}")

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

        print(f"Train set (orijinal): {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")

        # === Sınıf Dengesizliği Düzeltme Uygulaması (YALNIZCA EĞİTİM SETİNE) ===
        if self.imbalance_strategy == "smote":
            print("SMOTE uygulanıyor...")
            print(f"Train set dağılımı (SMOTE öncesi): {Counter(y_train_encoded)}")
            smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=self.smote_k_neighbors)
            X_train, y_train_encoded = smote.fit_resample(X_train, y_train_encoded)
            print(f"Train set dağılımı (SMOTE sonrası): {Counter(y_train_encoded)}")
            print(f"Train set boyutu (SMOTE sonrası): {X_train.shape[0]} samples")
        # 'class_weight' stratejisi modelin fit metodunda ele alınacaktır.

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

    def _get_xgboost_base_model(self):
        params = {
            "random_state": RANDOM_STATE,
            "eval_metric": "mlogloss",
            "use_label_encoder": False,
            "tree_method": "hist",
            "num_class": len(self.mbti_types),
            "objective": "multi:softmax",
        }
        if self.use_gpu:
            params["predictor"] = "gpu_predictor"
            params["gpu_id"] = 0

        # Sınıf ağırlıklarını uygulama
        class_weights = None
        if self.imbalance_strategy == "class_weight":
            # compute_class_weight, orijinal (kodlanmamış) etiketler yerine
            # kodlanmış etiketler üzerinde de çalışabilir, ancak genellikle
            # orijinal etiketlerin dağılımını yansıtmak daha doğru olur.
            # Burada y_train_encoded kullanıyoruz, bu da her bir kodlanmış sınıfın ağırlığını verir.
            weights = compute_class_weight(
                class_weight="balanced",
                classes=np.unique(
                    self.label_encoder.transform(self.label_encoder.classes_)
                ),  # Tüm olası sınıfların kodlanmış halleri
                y=self.label_encoder.inverse_transform(
                    y_train_encoded_for_weights
                ),  # Ağırlık hesaplaması için orijinal etiketlerin dağılımı
            )
            class_weights = {
                self.label_encoder.transform([cls])[0]: weight
                for cls, weight in zip(self.label_encoder.classes_, weights)
            }
            print(f"Hesaplanan sınıf ağırlıkları: {class_weights}")
            # XGBoost'ta doğrudan 'class_weight' parametresi yoktur.
            # 'scale_pos_weight' binary sınıflandırma içindir.
            # Multi-class için her örneğin ağırlığını `sample_weight` ile geçirmemiz gerekir.
            # Bunu `train_model` metodunda handle edeceğiz.

        return XGBClassifier(**params)

    def randomized_search_xgboost(self, X_train, y_train_encoded, X_val, y_val_encoded):
        print("\n16 MBTI tipi için RandomizedSearchCV başlatılıyor...")
        param_distributions = {
            "n_estimators": randint(100, 500),
            "max_depth": randint(3, 10),
            "learning_rate": uniform(0.01, 0.2),
            "subsample": uniform(0.7, 0.3),
            "colsample_bytree": uniform(0.7, 0.3),
        }

        xgb_model = self._get_xgboost_base_model()
        print(
            f"XGBClassifier {'GPU' if self.use_gpu else 'CPU'} ile başlatıldı: 16-tip sınıflandırma"
        )

        # Sınıf ağırlıklarını RandomizedSearchCV içinde kullanmak için
        sample_weights = None
        if self.imbalance_strategy == "class_weight":
            # y_train_encoded_for_weights burada _get_xgboost_base_model'de tanımlanmadığı için sorun çıkaracaktır.
            # Y_train'in orijinal kodlanmış hallerini kullanarak sample_weights oluşturmalıyız.
            # compute_class_weight fonksiyonu doğrudan etiket dizisi üzerinde çalışır.
            # Burada y_train_encoded zaten split_data'dan gelmekte.
            class_weights_arr = compute_class_weight(
                class_weight="balanced",
                classes=np.unique(
                    y_train_encoded
                ),  # Bu kısım önemlidir, SMOTE uygulanmamış orijinal y_train_encoded olmalı
                y=y_train_encoded,
            )
            class_weight_dict = dict(zip(np.unique(y_train_encoded), class_weights_arr))
            sample_weights = np.array(
                [class_weight_dict[label] for label in y_train_encoded]
            )
            print(
                f"RandomizedSearchCV için örnek ağırlıkları hesaplandı. İlk 5: {sample_weights[:5]}"
            )

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
            X_train,
            y_train_encoded,
            eval_set=[(X_val, y_val_encoded)],
            verbose=False,
            sample_weight=sample_weights,  # Sample weight RandomizedSearchCV fit metoduna iletildi
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

        sample_weights = None
        if self.imbalance_strategy == "class_weight":
            # compute_class_weight fonksiyonu burada SMOTE uygulanmamış orijinal y_train_encoded üzerinde çalışmalı
            # Aksi takdirde SMOTE'un zaten dengelediği dağılıma göre ağırlık hesaplar ki bu istenmez.
            # Bu nedenle, split_data'da y_train_encoded'ın SMOTE'dan önceki halini kullanmamız lazım,
            # ancak split_data metodunun dönüş değerleri SMOTE sonrası X_train ve y_train_encoded'ı temsil ediyor.
            # En iyi yaklaşım, class_weight'i doğrudan modelin parametresi olarak (eğer varsa) ayarlamak
            # veya compute_class_weight'i split_data öncesi orijinal label'lar üzerinde çalıştırmak.

            # Şimdilik, y_train_encoded_original gibi bir değişken tutmuyoruz.
            # Bu yüzden y_train_encoded (SMOTE sonrası hali olabilir) üzerinde ağırlık hesaplayacağız.
            # Eğer CLASS_IMBALANCE_STRATEGY = 'smote' ise 'class_weight' çalışmayacak, bu problem değil.

            # Y_train_encoded'ın SMOTE uygulanmamış halini almak için daha karmaşık bir state yönetimi gerekir.
            # Şimdilik, eğitilecek olan y_train_encoded'ın mevcut dağılımına göre ağırlık hesaplayalım.
            # Eğer 'smote' seçildiyse bu kısım çalışmayacak, 'class_weight' seçilirse bu y_train_encoded SMOTE'dan geçmemiş olur.
            class_weights_arr = compute_class_weight(
                class_weight="balanced",
                classes=np.unique(y_train_encoded),
                y=y_train_encoded,
            )
            class_weight_dict = dict(zip(np.unique(y_train_encoded), class_weights_arr))
            sample_weights = np.array(
                [class_weight_dict[label] for label in y_train_encoded]
            )
            print(
                f"Model eğitimi için örnek ağırlıkları hesaplandı. İlk 5: {sample_weights[:5]}"
            )

        if self.use_random_search:
            self.model = self.randomized_search_xgboost(
                X_train_scaled, y_train_encoded, X_val_scaled, y_val_encoded
            )
        else:
            print(
                "RandomizedSearchCV kullanılmadığı için sabit parametrelerle model eğitiliyor."
            )
            xgb_params = {
                "random_state": RANDOM_STATE,
                "eval_metric": "mlogloss",
                "use_label_encoder": False,
                "tree_method": "hist",
                "num_class": len(self.mbti_types),
                "objective": "multi:softmax",
                "n_estimators": 379,
                "max_depth": 7,
                "learning_rate": 0.06426980635477918,
                "subsample": 0.807025998008,
                "colsample_bytree": 0.8166031869,
            }
            if self.use_gpu:
                xgb_params["predictor"] = "gpu_predictor"
                xgb_params["gpu_id"] = 0

            self.model = XGBClassifier(**xgb_params)

            self.model.fit(
                X_train_scaled,
                y_train_encoded,
                eval_set=[(X_val_scaled, y_val_encoded)],
                verbose=False,
                sample_weight=sample_weights,  # Sample weight buraya da iletildi
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
            "imbalance_strategy": self.imbalance_strategy,  # Yeni
            "smote_k_neighbors": self.smote_k_neighbors,  # Yeni
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
        self.imbalance_strategy = model_data.get("imbalance_strategy", "none")  # Yeni
        self.smote_k_neighbors = model_data.get("smote_k_neighbors", 5)  # Yeni
        print(f"Model yüklendi: {filepath}")


def train_main():
    print("Veri yükleniyor...")

    # Analyzer sınıfını başlatırken config'den gelen parametreleri kullanın
    analyzer = MBTIXGBoostAnalyzer(
        use_random_search=USE_RANDOM_SEARCH,
        imbalance_strategy=CLASS_IMBALANCE_STRATEGY,
        smote_k_neighbors=SMOTE_K_NEIGHBORS,
    )

    if USE_BERT_EMBEDDINGS:
        feature_filepath = EMBEDDINGS_FILE_PATH_BERT
        print(f"BERT özellikleriyle birlikte veri yüklenecek: {feature_filepath}.npz")
    else:
        feature_filepath = EMBEDDINGS_FILE_PATH
        print(f"Yalnızca TF-IDF özellikleriyle veri yüklenecek: {feature_filepath}.npz")

    features, labels, feature_names = analyzer.load_data(f"{feature_filepath}.npz")
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

    analyzer.plot_confusion_matrix(
        y_test_encoded, test_pred, "Test Set Confusion Matrix (16 MBTI Types)"
    )
    analyzer.get_feature_importance(top_n=20)

    analyzer.save_model(XGBOOST_MODEL_PATH)


if __name__ == "__main__":
    train_main()
