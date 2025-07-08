# get_embeddings.py
import pandas as pd
import numpy as np
import re
import pickle
from typing import Dict, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import os
import sys

# BERT için gerekli kütüphaneler
from transformers import AutoTokenizer, AutoModel
import torch

# Add the parent directory to the path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import (
    MBTI_DATASET_PATH,
    EMBEDDINGS_FILE_PATH,
    EMBEDDINGS_FILE_PATH_BERT,
    TFIDF_VECTORIZER_PATH,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    USE_BERT_EMBEDDINGS,
    BERT_MODEL_NAME,
)

tqdm.pandas()


class MBTI_TFIDF_FeatureExtractor:
    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 2),
        use_bert: bool = False,  # Yeni parametre
        bert_model_name: str = "dbmdz/bert-base-turkish-cased",  # Yeni parametre
    ):
        print(
            f"TF-IDF vektörleştirici başlatılıyor (max_features={max_features}, ngram_range={ngram_range})..."
        )
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features, ngram_range=ngram_range
        )
        self.feature_names = []
        self.use_bert = use_bert  # Kullanım durumu
        self.bert_model = None
        self.bert_tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.use_bert:
            print(f"BERT modeli yükleniyor: {bert_model_name}...")
            try:
                self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
                self.bert_model = AutoModel.from_pretrained(bert_model_name).to(
                    self.device
                )
                print(f"BERT modeli {self.device} cihazına yüklendi.")
            except Exception as e:
                print(f"BERT modeli yüklenirken hata oluştu: {e}")
                self.use_bert = False  # Yükleme başarısız olursa BERT kullanımını kapat
                print("BERT embedding'leri devre dışı bırakıldı.")

        self.linguistic_feature_cols = [
            "word_count",
            "sentence_count",
            "avg_sentence_length",
            "exclamation_ratio",
            "question_ratio",
            "caps_ratio",
            "punctuation_diversity",
            "word_diversity",
            "future_tense_ratio",
            "past_tense_ratio",
            "personal_pronoun_ratio",
            "emotion_word_ratio",
            "social_word_ratio",
            "certainty_word_ratio",
            "uncertainty_word_ratio",
            "analytical_word_ratio",
        ]

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.lower()  # Tüm metni küçük harfe çevirin
        text = re.sub(r"http[s]?://\S+", "", text)  # URL'leri kaldırın
        text = re.sub(r"<[^>]+>", "", text)  # HTML etiketlerini kaldırın
        text = re.sub(
            r"[^a-z0-9\sğüşöçı]", "", text
        )  # Sadece küçük harf (Türkçe dahil), rakam ve boşlukları koru
        text = re.sub(r"\s+", " ", text)  # Birden fazla boşluğu tek boşluğa indir
        return text.strip()

    def extract_linguistic_features(self, text: str) -> Dict:
        if not text:
            return self._empty_linguistic_features()

        words = text.split()
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        word_count = len(words)
        sentence_count = len(sentences)
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0

        exclamation_count = text.count("!")
        question_count = text.count("?")

        uppercase_count = sum(1 for c in text if c.isupper())
        total_chars = len(text)
        caps_ratio = uppercase_count / total_chars if total_chars > 0 else 0

        punctuation = ".,;:!?-()[]{}\\'\""
        punct_diversity = len(set(c for c in text if c in punctuation))

        unique_words = len(set(word.lower() for word in words))
        word_diversity = unique_words / word_count if word_count > 0 else 0

        future_words = ["will", "going", "plan", "future", "tomorrow", "next", "soon"]
        past_words = ["was", "were", "had", "did", "yesterday", "before", "ago"]

        future_count = sum(text.lower().count(word) for word in future_words)
        past_count = sum(text.lower().count(word) for word in past_words)

        personal_pronouns = ["i", "me", "my", "mine", "myself"]
        personal_pronoun_count = sum(
            text.lower().count(word) for word in personal_pronouns
        )

        emotion_words = [
            "feel",
            "love",
            "hate",
            "happy",
            "sad",
            "angry",
            "excited",
            "worried",
        ]
        emotion_count = sum(text.lower().count(word) for word in emotion_words)

        social_words = ["we", "us", "our", "together", "friends", "people", "everyone"]
        social_count = sum(text.lower().count(word) for word in social_words)

        certainty_words = [
            "always",
            "never",
            "definitely",
            "absolutely",
            "certainly",
            "sure",
        ]
        certainty_count = sum(text.lower().count(word) for word in certainty_words)

        uncertainty_words = [
            "maybe",
            "perhaps",
            "might",
            "could",
            "possibly",
            "probably",
        ]
        uncertainty_count = sum(text.lower().count(word) for word in uncertainty_words)

        analytical_words = [
            "analyze",
            "think",
            "reason",
            "logic",
            "because",
            "therefore",
            "however",
        ]
        analytical_count = sum(text.lower().count(word) for word in analytical_words)

        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_sentence_length": avg_sentence_length,
            "exclamation_ratio": (
                exclamation_count / word_count if word_count > 0 else 0
            ),
            "question_ratio": question_count / word_count if word_count > 0 else 0,
            "caps_ratio": caps_ratio,
            "punctuation_diversity": punct_diversity,
            "word_diversity": word_diversity,
            "future_tense_ratio": future_count / word_count if word_count > 0 else 0,
            "past_tense_ratio": past_count / word_count if word_count > 0 else 0,
            "personal_pronoun_ratio": (
                personal_pronoun_count / word_count if word_count > 0 else 0
            ),
            "emotion_word_ratio": emotion_count / word_count if word_count > 0 else 0,
            "social_word_ratio": social_count / word_count if word_count > 0 else 0,
            "certainty_word_ratio": (
                certainty_count / word_count if word_count > 0 else 0
            ),
            "uncertainty_word_ratio": (
                uncertainty_count / word_count if word_count > 0 else 0
            ),
            "analytical_word_ratio": (
                analytical_count / word_count if word_count > 0 else 0
            ),
        }

    def _empty_linguistic_features(self) -> Dict:
        return {
            "word_count": 0,
            "sentence_count": 0,
            "avg_sentence_length": 0,
            "exclamation_ratio": 0,
            "question_ratio": 0,
            "caps_ratio": 0,
            "punctuation_diversity": 0,
            "word_diversity": 0,
            "future_tense_ratio": 0,
            "past_tense_ratio": 0,
            "personal_pronoun_ratio": 0,
            "emotion_word_ratio": 0,
            "social_word_ratio": 0,
            "certainty_word_ratio": 0,
            "uncertainty_word_ratio": 0,
            "analytical_word_ratio": 0,
        }

    def _get_bert_embedding(self, text: str) -> np.ndarray:
        """Metinden BERT embedding'i çıkarır."""
        if not self.use_bert or not self.bert_model or not self.bert_tokenizer:
            return np.array([])  # BERT kullanılmıyorsa boş dizi döndür

        try:
            inputs = self.bert_tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",  # Doldurma
                truncation=True,  # Kesme
                max_length=512,  # Maksimum token uzunluğu
            ).to(self.device)

            with torch.no_grad():
                outputs = self.bert_model(**inputs)

            # Son gizli durum çıktısını al
            # Burada CLS token embedding'i (ilk token) veya tüm tokenlerin ortalaması kullanılabilir
            # Genellikle CLS tokenı cümle seviyesi temsili için kullanılır
            sentence_embedding = (
                outputs.last_hidden_state[:, 0, :].cpu().numpy()
            )  # CLS token embedding'i

            # Veya tüm token embedding'lerinin ortalaması
            # sentence_embedding = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()

            return sentence_embedding.flatten()  # 2D'den 1D'ye düzleştir
        except Exception as e:
            print(f"BERT embedding çıkarılırken hata oluştu: {e}")
            # Hata durumunda sıfırlardan oluşan bir embedding döndür (boyut modelin gizli boyutuna bağlı)
            # Örneğin, dbmdz/bert-base-turkish-cased için gizli boyut 768'dir.
            return np.zeros(self.bert_model.config.hidden_size)

    def extract_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        print("Metinler temizleniyor...")
        data["cleaned_posts"] = data["posts"].progress_apply(self.clean_text)

        print("TF-IDF features çıkarılıyor ve vektörleştirici eğitiliyor...")
        tfidf_features = self.tfidf_vectorizer.fit_transform(
            data["cleaned_posts"]
        ).toarray()
        tfidf_feature_names = self.tfidf_vectorizer.get_feature_names_out().tolist()

        print("Linguistic features çıkarılıyor...")
        linguistic_features_list = (
            data["cleaned_posts"]
            .progress_apply(self.extract_linguistic_features)
            .tolist()
        )
        linguistic_df = pd.DataFrame(linguistic_features_list)
        linguistic_feature_names = [f"ling_{col}" for col in linguistic_df.columns]
        linguistic_array = linguistic_df.values
        linguistic_array = np.nan_to_num(linguistic_array)  # NaN değerleri sıfıra çevir

        # Özellikleri birleştirme
        feature_matrix_parts = [tfidf_features, linguistic_array]
        all_feature_names = tfidf_feature_names + linguistic_feature_names

        if self.use_bert:
            print("BERT features çıkarılıyor...")
            bert_embeddings_list = (
                data["cleaned_posts"].progress_apply(self._get_bert_embedding).tolist()
            )
            # Her bir embedding'in boyutunun aynı olduğundan emin ol (modelin gizli boyutu)
            # Eğer boş liste dönerse, sıfırlardan oluşan bir array ile doldur
            if not bert_embeddings_list or len(bert_embeddings_list[0]) == 0:
                print(
                    "Uyarı: BERT embedding'leri boş veya çıkarılamadı. BERT özellikleri dahil edilmeyecek."
                )
                self.use_bert = False  # Hata durumunda BERT kullanımını kapat
            else:
                bert_embeddings_array = np.array(bert_embeddings_list)
                feature_matrix_parts.append(bert_embeddings_array)
                bert_feature_names = [
                    f"bert_{i}" for i in range(bert_embeddings_array.shape[1])
                ]
                all_feature_names += bert_feature_names

        print("Features birleştiriliyor...")
        feature_matrix = np.hstack(feature_matrix_parts)
        self.feature_names = all_feature_names

        print(f"Toplam {feature_matrix.shape[1]} feature çıkarıldı")
        print(f"- TF-IDF features: {len(tfidf_feature_names)}")
        print(f"- Linguistic features: {len(linguistic_feature_names)}")
        if self.use_bert:
            print(
                f"- BERT features: {bert_embeddings_array.shape[1]}"
            )  # Yalnızca başarılıysa yazdır

        return feature_matrix, self.feature_names

    def save_features(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        feature_names: List[str],
        filepath: str,  # Şimdi bu filepath, config'den gelen EMBEDDINGS_FILE_PATH_BERT veya EMBEDDINGS_FILE_PATH olacak
    ):
        np.savez(
            f"{filepath}.npz",
            features=features,
            labels=labels,
            feature_names=np.array(feature_names),
        )
        feature_df = pd.DataFrame(features, columns=feature_names)
        feature_df["mbti_type"] = labels
        feature_df.to_csv(f"{filepath}.csv", index=False)

        # TF-IDF vektörleştiriciyi sadece TF-IDF kullanılıyorsa kaydedin
        # veya her zaman kaydedip farklı bir isimle ayırın.
        # Bu durumda, BERT'siz versiyon için de TF-IDF vektörleştiriciyi ayrı kaydediyoruz.
        with open(
            f"{TFIDF_VECTORIZER_PATH}", "wb"
        ) as f:  # TFIDF_VECTORIZER_PATH'i kullan
            pickle.dump(self.tfidf_vectorizer, f)

        print(f"Features kaydedildi: {filepath}.npz, {filepath}.csv")
        print(f"TF-IDF vektörleştirici kaydedildi: {TFIDF_VECTORIZER_PATH}")
        print(f"Feature shape: {features.shape}")

    def load_features(self, filepath: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        data = np.load(f"{filepath}.npz", allow_pickle=True)
        features = data["features"]
        labels = data["labels"]
        feature_names = data["feature_names"].tolist()
        try:
            with open(
                f"{TFIDF_VECTORIZER_PATH}", "rb"
            ) as f:  # Yüklerken de TFIDF_VECTORIZER_PATH'i kullan
                self.tfidf_vectorizer = pickle.load(f)
            print(f"TF-IDF vektörleştirici yüklendi: {TFIDF_VECTORIZER_PATH}")
        except FileNotFoundError:
            print(
                f"Uyarı: {TFIDF_VECTORIZER_PATH} bulunamadı. TF-IDF vektörleştirici yeniden yüklenmedi."
            )
        return features, labels, feature_names


def get_embeddings_main(
    file_path: str,
    output_path_tfidf: str,
    output_path_bert: str,
    use_bert_embeddings: bool,
):
    print("Veri yükleniyor...")
    data = pd.read_csv(file_path)

    extractor = MBTI_TFIDF_FeatureExtractor(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        use_bert=use_bert_embeddings,  # config'den gelen değeri kullan
        bert_model_name=BERT_MODEL_NAME,  # config'den gelen değeri kullan
    )
    features, feature_names = extractor.extract_features(data)
    labels = data["type"].values

    if use_bert_embeddings:
        extractor.save_features(features, labels, feature_names, output_path_bert)
    else:
        extractor.save_features(features, labels, feature_names, output_path_tfidf)

    print("\nİşlem tamamlandı!")
    print(f"Feature matrix shape: {features.shape}")  # Genel feature shape
    print(f"Unique MBTI types: {np.unique(labels)}")


if __name__ == "__main__":
    # TF-IDF + BERT özelliklerini çıkarmak için
    get_embeddings_main(
        MBTI_DATASET_PATH,
        EMBEDDINGS_FILE_PATH,
        EMBEDDINGS_FILE_PATH_BERT,
        USE_BERT_EMBEDDINGS,
    )

    # Yalnızca TF-IDF özelliklerini çıkarmak isterseniz (USE_BERT_EMBEDDINGS = False olarak ayarlıysa bu çalışır)
    # Veya isterseniz USE_BERT_EMBEDDINGS'i True yapıp ayrı bir çağrı da yapabilirsiniz
    # get_embeddings_main(MBTI_DATASET_PATH, EMBEDDINGS_FILE_PATH, EMBEDDINGS_FILE_PATH_BERT, False)
