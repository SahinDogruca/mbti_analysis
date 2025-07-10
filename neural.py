# neural.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import joblib
import warnings
import os
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List
from tqdm.auto import tqdm
from pathlib import Path

# PyTorch kütüphaneleri
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Import global variables directly from config
from config import (
    DATA_DIR,
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    PLOTS_DIR,
    EMBEDDINGS_FILE_PATH,
    EMBEDDINGS_FILE_PATH_BERT,
    TEST_SIZE,
    VAL_SIZE,
    RANDOM_STATE,
    NEURAL_NETWORK_LEARNING_RATE,
    NEURAL_NETWORK_EPOCHS,
    NEURAL_NETWORK_BATCH_SIZE,
    USE_BERT_EMBEDDINGS,  # Added for conditional data loading
)

warnings.filterwarnings("ignore")

# Cihaz ayarı (GPU varsa CUDA kullan, yoksa CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Eğitim için kullanılacak cihaz: {device}")


# --- Yapay Sinir Ağı Modeli Sınıfı ---
class MBTINeuralNetwork(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, model_type: str = "cnn"):
        super(MBTINeuralNetwork, self).__init__()
        self.model_type = model_type

        if model_type == "cnn":
            # Basit bir 1D CNN katmanı
            # Giriş: (batch_size, input_dim) -> (batch_size, 1, input_dim)
            self.cnn_feature_extractor = nn.Sequential(
                nn.Conv1d(
                    in_channels=1, out_channels=64, kernel_size=3, padding=1
                ),  # (batch_size, 64, input_dim)
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),  # (batch_size, 64, input_dim/2)
                nn.Conv1d(
                    in_channels=64, out_channels=128, kernel_size=3, padding=1
                ),  # (batch_size, 128, input_dim/2)
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),  # (batch_size, 128, input_dim/4)
                nn.Flatten(),  # Düzleştirme
            )
            # CNN çıkış boyutunu hesapla
            # Dinamik olarak hesaplamak için dummy_input kullanıldı
            dummy_input = torch.randn(1, 1, input_dim)
            cnn_output_dim = self.cnn_feature_extractor(dummy_input).shape[1]
            self.classifier = nn.Linear(cnn_output_dim, num_classes)

        elif model_type == "lstm":
            # Basit bir LSTM katmanı
            # Giriş: (batch_size, sequence_length, input_size)
            # Burada her bir feature'ı bir zaman adımı gibi ele alıyoruz.
            self.lstm_feature_extractor = nn.LSTM(
                input_size=1, hidden_size=128, num_layers=2, batch_first=True
            )
            self.classifier = nn.Linear(128, num_classes)
        else:
            # Sadece doğrusal katmanlar (MLP)
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes),
            )

    def forward(self, x):
        if self.model_type == "cnn":
            # CNN için girişi (batch_size, 1, input_dim) şekline getir
            x = x.unsqueeze(1)
            x = self.cnn_feature_extractor(x)
        elif self.model_type == "lstm":
            # LSTM için girişi (batch_size, sequence_length, input_size) şekline getir
            x = x.unsqueeze(-1)  # (batch_size, input_dim, 1)
            _, (h_n, c_n) = self.lstm_feature_extractor(x)
            x = h_n[-1, :, :]  # Son gizli durumu al
        # MLP için x olduğu gibi kalır

        x = self.classifier(x)
        return x


# --- MBTINNAnalyzer Sınıfı ---
class MBTINNAnalyzer:
    def __init__(self):
        self.model: Optional[MBTINeuralNetwork] = None
        self.scaler: Optional[StandardScaler] = None
        self.label_encoder = LabelEncoder()
        self.feature_names: List[str] = []
        self.mbti_types: List[str] = []
        self.device = device  # Cihazı Analyzer sınıfına ekle

    def load_data(
        self,
        filepath: str,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """NPZ dosyasından veriyi yükle"""
        print(f"Veri yükleniyor: {filepath}...")

        if not Path(filepath).exists():
            raise FileNotFoundError(
                f"Özellik dosyası bulunamadı: {filepath}. Lütfen önce `get_embeddings.py`'yi çalıştırın."
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
        test_size: float = TEST_SIZE,
        val_size: float = VAL_SIZE,
        random_state: int = RANDOM_STATE,
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

    def train_model(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        y_train_encoded: np.ndarray,
        y_val_encoded: np.ndarray,
        model_type: str = "cnn",  # 'cnn', 'lstm' veya 'mlp'
        learning_rate: float = NEURAL_NETWORK_LEARNING_RATE,
        epochs: int = NEURAL_NETWORK_EPOCHS,
        batch_size: int = NEURAL_NETWORK_BATCH_SIZE,
    ) -> None:
        """
        Yapay Sinir Ağı modelini eğitir.
        """
        print("\n=== YAPAY SİNİR AĞI MODEL EĞİTİMİ BAŞLATIYOR ===")

        X_train_scaled, X_val_scaled, _ = self.scale_features(
            X_train,
            X_val,
            X_val,  # Pass X_val twice as test is not needed for scaling here
        )

        input_dim = X_train_scaled.shape[1]
        num_classes = len(self.mbti_types)

        self.model = MBTINeuralNetwork(input_dim, num_classes, model_type).to(
            self.device
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # DataLoader oluştur
        train_dataset = TensorDataset(
            torch.tensor(X_train_scaled), torch.tensor(y_train_encoded)
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val_scaled), torch.tensor(y_val_encoded)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        print(f"Model tipi: {model_type}")
        print(
            f"Öğrenme oranı: {learning_rate}, Epochs: {epochs}, Batch boyutu: {batch_size}"
        )

        # Eğitim döngüsü
        for epoch in range(epochs):
            self.model.train()  # Modeli eğitim moduna al
            running_loss = 0.0
            correct_train = 0
            total_train = 0

            for inputs, labels in tqdm(
                train_loader, desc=f"Epoch {epoch+1}/{epochs} (Eğitim)"
            ):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

            epoch_loss = running_loss / len(train_dataset)
            epoch_acc = correct_train / total_train

            # Doğrulama (Validation)
            self.model.eval()  # Modeli değerlendirme moduna al
            val_loss = 0.0
            correct_val = 0
            total_val = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()

            val_epoch_loss = val_loss / len(val_dataset)
            val_epoch_acc = correct_val / total_val

            print(
                f"Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, "
                f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}"
            )

        print("Model eğitimi tamamlandı.")

    def predict_mbti(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """MBTI tiplerini tahmin et"""
        self.model.eval()  # Modeli değerlendirme moduna al
        X_scaled = self.scaler.transform(X) if self.scaler else X
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            predicted_encoded = torch.argmax(outputs, dim=1).cpu().numpy()

        predicted_mbti_types = self.label_encoder.inverse_transform(predicted_encoded)

        confidences = np.array(
            [probabilities[i, pred] for i, pred in enumerate(predicted_encoded)]
        )

        return predicted_mbti_types, confidences

    def evaluate_model(
        self, X_data: np.ndarray, y_true_encoded: np.ndarray, dataset_name: str = "Test"
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """Model performansını değerlendir"""
        print(f"\n=== {dataset_name.upper()} SETİ DEĞERLENDİRMESİ ===")

        predicted_mbti, confidences = self.predict_mbti(X_data)

        y_true_original = self.label_encoder.inverse_transform(y_true_encoded)

        overall_acc = accuracy_score(y_true_original, predicted_mbti)
        print(f"\nOverall MBTI Accuracy: {overall_acc:.4f}")

        print("\n--- Detailed Classification Report ---")
        print(
            classification_report(
                y_true_original,
                predicted_mbti,
                target_names=self.mbti_types,
                zero_division=0,
            )
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

    def save_model(self, model_prefix: str, data_type: str, model_type: str):
        """Modeli kaydet"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{model_prefix}_{data_type}_{model_type}_{timestamp}.pkl"
        model_filepath = Path(os.path.join(MODEL_DIR, model_filename))

        # PyTorch modelini ve diğer bileşenleri kaydet
        model_data = {
            "model_state_dict": self.model.state_dict(),
            "scaler": self.scaler,
            "label_encoder": self.label_encoder,
            "feature_names": self.feature_names,
            "mbti_types": self.mbti_types,
            "input_dim": (
                self.model.classifier[0].in_features
                if isinstance(self.model.classifier, nn.Sequential)
                and len(self.model.classifier) > 0
                else (
                    self.model.classifier.in_features
                    if hasattr(self.model.classifier, "in_features")
                    else None
                )
            ),
            "num_classes": len(self.mbti_types),
            "model_type": self.model.model_type,  # Model tipini kaydet
        }

        joblib.dump(model_data, model_filepath)
        print(f"Model kaydedildi: {model_filepath}")

    def load_model(self, filepath: Path):
        """Modeli yükle"""
        if not filepath.exists():
            raise FileNotFoundError(f"Model dosyası bulunamadı: {filepath}")

        model_data = joblib.load(filepath)

        self.scaler = model_data["scaler"]
        self.label_encoder = model_data["label_encoder"]
        self.feature_names = model_data["feature_names"]
        self.mbti_types = model_data["mbti_types"]
        input_dim = model_data["input_dim"]
        num_classes = model_data["num_classes"]
        model_type = model_data.get("model_type", "mlp")  # Varsayılan olarak mlp

        self.model = MBTINeuralNetwork(input_dim, num_classes, model_type).to(
            self.device
        )
        self.model.load_state_dict(model_data["model_state_dict"])
        self.model.eval()  # Yükledikten sonra değerlendirme moduna al

        print(f"Model yüklendi: {filepath}")


# --- Ana Pipeline Fonksiyonu (xgboost_train.py'ye benzer şekilde refaktör edildi) ---
def neural_train_main(
    data_type: str = "mbti_embeddings_tfidf",  # Bu varsayılan değeri değiştirmeyeceğiz.
    model_type: str = "cnn",  # 'cnn', 'lstm' veya 'mlp'
):
    """Belirtilen data_type ile MBTI Yapay Sinir Ağı analiz pipeline'ını çalıştırır."""
    print(f"\n{'='*80}")
    print(
        f"--- MBTI Yapay Sinir Ağı Model Eğitimi ({data_type}, Model Tipi: {model_type}) ---"
    )
    print(f"{'='*80}\n")

    analyzer = MBTINNAnalyzer()

    # USE_BERT_EMBEDDINGS ayarına göre doğru özellik dosya yolunu seçin
    if USE_BERT_EMBEDDINGS and data_type == "mbti_embeddings_tfidf_bert":
        feature_filepath = f"{EMBEDDINGS_FILE_PATH_BERT}.npz"
        print(f"BERT özellikleriyle birlikte veri yüklenecek: {feature_filepath}")
    elif data_type == "mbti_embeddings_tfidf":
        feature_filepath = f"{EMBEDDINGS_FILE_PATH}.npz"
        print(f"Yalnızca TF-IDF özellikleriyle veri yüklenecek: {feature_filepath}")
    else:
        raise ValueError(
            f"Geçersiz data_type: {data_type}. Lütfen 'mbti_embeddings_tfidf' veya 'mbti_embeddings_tfidf_bert' kullanın."
        )

    features, labels, feature_names = analyzer.load_data(filepath=feature_filepath)
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

    analyzer.train_model(
        X_train,
        X_val,
        y_train_encoded,
        y_val_encoded,
        model_type=model_type,
        learning_rate=NEURAL_NETWORK_LEARNING_RATE,
        epochs=NEURAL_NETWORK_EPOCHS,
        batch_size=NEURAL_NETWORK_BATCH_SIZE,
    )

    # Doğrulama Seti Üzerinde Değerlendirme
    print("\n--- Doğrulama Seti Performansı ---")
    val_pred, val_conf, val_results = analyzer.evaluate_model(
        analyzer.scaler.transform(X_val), y_val_encoded, "Doğrulama"
    )  # Use scaled data for evaluation
    # Doğrulama Seti için Confusion Matrix çizimini kaydet
    timestamp_val = datetime.now().strftime("%Y%m%d_%H%M%S")
    analyzer.plot_confusion_matrix(
        y_val_encoded,
        val_pred,
        title=f"Doğrulama Seti Karışıklık Matrisi ({data_type}, {model_type})",
        save_path=Path(
            os.path.join(
                PLOTS_DIR,
                f"confusion_matrix_val_{data_type}_{model_type}_{timestamp_val}.png",
            )
        ),
    )

    # Test Seti Üzerinde Değerlendirme
    print("\n--- Test Seti Performansı ---")
    test_pred, test_conf, test_results = analyzer.evaluate_model(
        analyzer.scaler.transform(X_test), y_test_encoded, "Test"
    )  # Use scaled data for evaluation
    # Test Seti için Confusion Matrix çizimini kaydet
    timestamp_test = datetime.now().strftime("%Y%m%d_%H%M%S")
    analyzer.plot_confusion_matrix(
        y_test_encoded,
        test_pred,
        title=f"Test Seti Karışıklık Matrisi ({data_type}, {model_type})",
        save_path=Path(
            os.path.join(
                PLOTS_DIR,
                f"confusion_matrix_test_{data_type}_{model_type}_{timestamp_test}.png",
            )
        ),
    )

    # Modeli veri tipine ve model tipine göre benzersiz bir önekle kaydet
    analyzer.save_model(
        model_prefix="mbti_neural_network_model",
        data_type=data_type,
        model_type=model_type,
    )

    print(f"\n=== ÖRNEK TAHMİN ({data_type}, {model_type}) ===")
    sample_indices = np.random.choice(
        len(analyzer.scaler.transform(X_test)), 5, replace=False
    )
    sample_features = analyzer.scaler.transform(X_test)[sample_indices]
    sample_true_encoded = y_test_encoded[sample_indices]
    sample_true_original = analyzer.label_encoder.inverse_transform(sample_true_encoded)

    sample_pred, sample_conf = analyzer.predict_mbti(sample_features)

    for i in range(5):
        print(
            f"Örnek {i+1}: True={sample_true_original[i]}, Pred={sample_pred[i]}, "
            f"Confidence={sample_conf[i]:.3f}"
        )

    print(
        f"\nAnaliz tamamlandı for {data_type} with {model_type} model! Model ve sonuçlar kaydedildi."
    )
    return analyzer


if __name__ == "__main__":
    # CNN modeli ile çalıştırma
    neural_train_main(data_type="mbti_embeddings_tfidf", model_type="cnn")

    print("\n" + "=" * 80)
    print("Tüm Yapay Sinir Ağı eğitim ve değerlendirme işlemleri tamamlandı.")
    print("=" * 80 + "\n")

    # --- Örnek: Kaydedilen Modeli Yükleme ---
    print("\n--- Kaydedilen Modeli Yükleme Örneği ---")
    try:
        print("\n'mbti_embeddings_tfidf' CNN modeli yükleniyor...")
        loaded_analyzer = MBTINNAnalyzer()
        # Modeli yüklemek için en son kaydedilen model dosyasını bul
        # Bu, timestamp içeren dosya adlandırma kuralına uygun olmalıdır.
        # Örneğin, MODEL_DIR içindeki en son dosyayı bulmak için:
        model_files = list(
            Path(MODEL_DIR).glob(
                "mbti_neural_network_model_mbti_embeddings_tfidf_cnn_*.pkl"
            )
        )
        if model_files:
            latest_model_file = max(model_files, key=os.path.getctime)
            loaded_analyzer.load_model(filepath=latest_model_file)
            print(f"Model ({latest_model_file.name}) başarıyla yüklendi.")
            print(
                f"Yüklenen modelin eğitiminde kullanılan özellik sayısı: {len(loaded_analyzer.feature_names)}"
            )
        else:
            print("Yüklenecek bir model bulunamadı.")

    except FileNotFoundError as e:
        print(f"Model yüklenirken hata oluştu: {e}")
    except Exception as e:
        print(f"Beklenmeyen bir hata oluştu: {e}")
