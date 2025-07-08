import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from typing import Dict, List, Tuple, Any, Optional
from nltk.stem import WordNetLemmatizer
from tqdm.auto import tqdm
import os
from datetime import datetime
from pathlib import Path

# Config sınıfını ayrı bir dosyadan (config.py) import ediyoruz
# Bu satırın, config.py dosyanızın konumuna göre ayarlanması gerekebilir.
# Örneğin, eğer config.py aynı dizindeyse bu şekilde kullanabilirsiniz:
from config import Config  # config.py dosyanızın adını ve sınıfın adını kontrol edin

# Veya, eğer config.py farklı bir alt dizindeyse (örneğin 'utils' klasöründe):
# from utils.config import Config

# NLTK verilerini indirme (bir kez çalıştırmak yeterlidir)
# import nltk
# nltk.download('wordnet')


# --- Lemmatizer Sınıfı (Değişiklik Yok) ---
class Lemmatizer(object):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def __call__(self, sentence):
        if not isinstance(sentence, str):
            return []
        return [
            self.lemmatizer.lemmatize(word)
            for word in sentence.split()
            if len(word) > 2
        ]


# --- MBTIFeatureExtractor Sınıfı (Değişiklikler Burada) ---
class MBTIFeatureExtractor:
    def __init__(self, config_instance: Config):
        self.config = config_instance  # Config objesi burada alınıyor
        self.tfidf_vectorizer = None
        self.feature_names = []

    def clean_text(self, text: str, remove_mbti_types: bool = False) -> str:
        """Metni temizle ve normalize et"""
        if not isinstance(text, str):
            return ""

        text = text.lower()

        # URL'leri temizle
        text = re.sub(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            "",
            text,
        )

        # HTML etiketlerini temizle
        text = re.sub(r"<[^>]+>", "", text)

        # Çoklu tekrar eden harfli kelimeleri kaldır
        text = re.sub(r"\b\w*([a-z])\1{2,}\w*\b", " ", text)

        # Çok kısa veya çok uzun kelimeleri kaldır
        text = re.sub(r"\b\w{0,3}\b", " ", text)
        text = re.sub(r"\b\w{30,}\b", " ", text)

        # MBTI Kişilik Kelimelerini Kaldır
        if remove_mbti_types:
            pers_types = [
                "INFP",
                "INFJ",
                "INTP",
                "INTJ",
                "ENTP",
                "ENFP",
                "ISTP",
                "ISFP",
                "ENTJ",
                "ISTJ",
                "ENFJ",
                "ISFJ",
                "ESTP",
                "ESFP",
                "ESFJ",
                "ESTJ",
            ]
            pers_types = [p.lower() for p in pers_types]
            p = re.compile(r"\b(" + "|".join(pers_types) + r")\b")
            text = p.sub(" ", text)

        # Sadece alfanümerik karakterleri ve boşlukları koru
        text = re.sub(r"[^0-9a-z\s]", " ", text)

        # Oluşan fazla boşlukları temizle
        text = re.sub(r"\s+", " ", text).strip()

        return text.strip()

    def extract_linguistic_features(self, text: str) -> Dict:
        """Metinden linguistic features çıkar"""
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

        punctuation = ".,;:!?-()[]{}\"'"
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
        """Boş metin için varsayılan linguistic features"""
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

    def extract_features(
        self, data: pd.DataFrame, remove_mbti_types: bool = False
    ) -> Tuple[np.ndarray, List[str]]:
        """Ana feature extraction fonksiyonu"""

        print("Metinler temizleniyor...")
        tqdm.pandas(desc="Metinleri Temizleme")
        data["cleaned_posts"] = data["posts"].progress_apply(
            lambda x: self.clean_text(x, remove_mbti_types)
        )

        print("TF-IDF özellikler çıkarılıyor...")
        # TF-IDF parametrelerini config'den al
        tfidf_params = self.config.get_embeddings_config()
        ngram_range_config = tfidf_params.get(
            "ngram_range", [1, 3]
        )  # Varsayılan değerler

        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=tfidf_params.get("max_features", 5000),
            ngram_range=(ngram_range_config[0], ngram_range_config[1]),
            min_df=tfidf_params.get("min_df", 0.01),
            max_df=tfidf_params.get("max_df", 0.95),
            stop_words="english",
            lowercase=False,
            tokenizer=Lemmatizer(),
        )

        tfidf_features = self.tfidf_vectorizer.fit_transform(data["cleaned_posts"])
        tfidf_feature_names = [
            f"tfidf_{name}" for name in self.tfidf_vectorizer.get_feature_names_out()
        ]

        print("Linguistik özellikler çıkarılıyor...")
        linguistic_features_list = []
        for text in tqdm(data["cleaned_posts"], desc="Linguistik Özellikleri Çıkarma"):
            linguistic_features_list.append(self.extract_linguistic_features(text))

        linguistic_df = pd.DataFrame(linguistic_features_list)
        linguistic_feature_names = [f"ling_{col}" for col in linguistic_df.columns]

        print("Özellikler birleştiriliyor...")
        tfidf_dense = tfidf_features.toarray()
        linguistic_array = linguistic_df.values
        linguistic_array = np.nan_to_num(linguistic_array)

        feature_matrix = np.hstack([tfidf_dense, linguistic_array])
        self.feature_names = tfidf_feature_names + linguistic_feature_names

        print(f"Toplam {feature_matrix.shape[1]} özellik çıkarıldı")
        print(f"- TF-IDF özellikleri: {len(tfidf_feature_names)}")
        print(f"- Linguistik özellikler: {len(linguistic_feature_names)}")

        return feature_matrix, self.feature_names

    def save_features(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        feature_names: List[str],
        data_type: str,  # data_type artık zorunlu
        version: Optional[str] = None,
    ):
        """Özellikleri kaydet"""
        # Özellikler ve vektörleyici için mevcut versiyonu kaydet
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Config'den yolları al
        npz_filepath = self.config.get_processed_data_path(
            data_type, version, extension=".npz"
        )
        csv_filepath = self.config.get_processed_data_path(
            data_type, version, extension=".csv"
        )
        # Vektörleyici için ayrı bir yol (genellikle aynı data_type ve versiyon ile ilişkilidir)
        vectorizer_filepath = self.config.get_processed_data_path(
            f"{data_type}_vectorizer", version, extension=".pkl"
        )

        # Numpy array olarak kaydet
        np.savez(
            npz_filepath,
            features=features,
            labels=labels,
            feature_names=np.array(feature_names),
        )

        # CSV olarak da kaydet (daha okunabilir)
        feature_df = pd.DataFrame(features, columns=feature_names)
        feature_df["mbti_type"] = labels
        feature_df.to_csv(csv_filepath, index=False)

        # TF-IDF vektörleyiciyi kaydet
        with open(vectorizer_filepath, "wb") as f:
            pickle.dump(self.tfidf_vectorizer, f)

        print(f"Özellikler kaydedildi: {npz_filepath}, {csv_filepath}")
        print(f"Vektörleyici kaydedildi: {vectorizer_filepath}")
        print(f"Özellik boyutu: {features.shape}")

    def load_features(
        self, data_type: str, version: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Kaydedilmiş özellikleri yükle"""

        # Eğer spesifik bir versiyon verilmemişse en son kaydedileni bul
        if version is None:
            npz_filepath = self.config.get_latest_processed_data_path(
                data_type, extension=".npz"
            )
            if npz_filepath is None:
                raise FileNotFoundError(
                    f"'{data_type}' tipi için en son işlenmiş veri bulunamadı."
                )
            # DİKKAT: BURADAKİ DEĞİŞİKLİK
            # Dosya adından tam versiyon bilgisini çıkar (örneğin: "mbti_features_20250708_131702" -> "20250708_131702")
            # stem: "mbti_features_20250708_131702"
            # split("_") -> ["mbti", "features", "20250708", "131702"]
            # join ile son iki parçayı birleştiriyoruz
            parts = npz_filepath.stem.split("_")
            # data_type'ın kendisi underscore içerebileceği için, version'ı son iki parçadan almak daha güvenli.
            # Örneğin "mbti_features_without_types_20250708_131702" için son iki parça "20250708_131702" olacaktır.
            if len(parts) >= 2 and re.match(r"\d{8}_\d{6}", "_".join(parts[-2:])):
                version = "_".join(parts[-2:])  # Son iki parçayı birleştir
            else:
                # Beklenenden farklı bir dosya adı formatıysa hata fırlat veya varsayılan atama yap
                raise ValueError(f"Beklenmeyen dosya adı formatı: {npz_filepath.stem}")
        else:
            npz_filepath = self.config.get_processed_data_path(
                data_type, version, extension=".npz"
            )

        # Vektörleyici dosya yolunu belirle
        vectorizer_filepath = self.config.get_processed_data_path(
            f"{data_type}_vectorizer", version, extension=".pkl"
        )

        if not npz_filepath.exists():
            raise FileNotFoundError(f"Özellikler dosyası bulunamadı: {npz_filepath}")
        if not vectorizer_filepath.exists():
            raise FileNotFoundError(
                f"Vektörleyici dosyası bulunamadı: {vectorizer_filepath}"
            )

        data_loaded = np.load(npz_filepath, allow_pickle=True)
        features = data_loaded["features"]
        labels = data_loaded["labels"]
        feature_names = data_loaded["feature_names"].tolist()

        with open(vectorizer_filepath, "rb") as f:
            self.tfidf_vectorizer = pickle.load(f)

        print(f"Özellikler yüklendi: {npz_filepath}")
        print(f"Vektörleyici yüklendi: {vectorizer_filepath}")
        return features, labels, feature_names


# --- Veri İşleme Fonksiyonu ---
def process_mbti_data(config_instance: Config, remove_mbti_types: bool = False):
    """MBTI datasını işle ve özelliklerini çıkar"""

    print("Veri yükleniyor...")
    # Ham veri yolunu config'den al
    file_path = config_instance.get_path("raw_data_file")
    data = pd.read_csv(file_path)

    extractor = MBTIFeatureExtractor(config_instance)
    features, feature_names = extractor.extract_features(data, remove_mbti_types)
    labels = data["type"].values

    # remove_mbti_types'a göre data_type belirle
    if remove_mbti_types:
        data_type_suffix = "with_types"
    else:
        data_type_suffix = "without_types"
    save_data_type = f"mbti_features_{data_type_suffix}"

    # Özellikleri ve vektörleyiciyi kaydet
    extractor.save_features(features, labels, feature_names, data_type=save_data_type)

    return features, labels, feature_names


# --- Kullanım Örneği ---
if __name__ == "__main__":
    # Eğer wordnet'i indirmediyseniz, aşağıdaki iki satırı yorumdan kaldırıp bir kez çalıştırın:
    # import nltk
    # nltk.download('wordnet')

    # Config objesinin bir örneğini oluştur
    # Bu, config.py dosyanızın içindeki Config sınıfını kullanacaktır.
    current_config = Config()  # Varsayılan olarak 'config.yaml' dosyasını arar

    # config.yaml dosyasında 'paths.raw_data_file' yolunun doğru olduğundan emin olun.
    # Örn: paths: raw_data_file: data/mbti_1.csv
    print(f"Ham veri yolu (config'den): {current_config.get_path('raw_data_file')}")
    print(
        f"TF-IDF Max Features (config'den): {current_config.get_embeddings_config().get('max_features')}"
    )

    print("\n" + "--- MBTI Tiplerini KALDIRMADAN verileri işleme ---" + "\n")
    features_no_mbti, labels_no_mbti, feature_names_no_mbti = process_mbti_data(
        current_config, remove_mbti_types=False
    )
    print("\nİşlem tamamlandı (MBTI tipleri çıkarılmadı)!")
    print(f"Özellik matrisi boyutu: {features_no_mbti.shape}")
    print(f"Benzersiz MBTI tipleri: {np.unique(labels_no_mbti)}")

    print("\n" + "=" * 80 + "\n")  # Ayrım için

    print("\n" + "--- MBTI Tiplerini KALDIRARAK verileri işleme ---" + "\n")
    features_with_mbti, labels_with_mbti, feature_names_with_mbti = process_mbti_data(
        current_config, remove_mbti_types=True
    )
    print(
        "\nİşlem tamamlandı (MBTI tipleri çıkarıldı)! (Bu durumda `remove_mbti_types=True` ise, metinden MBTI tipleri kaldırılır ancak dosya adına `with_types` eklenir, bu bir tutarsızlık olabilir. Eğer `remove_mbti_types=True` iken `without_types` olarak kaydedilmesini istiyorsanız, `data_type_suffix` mantığını tersine çevirmeniz gerekir.)"
    )
    print(f"Özellik matrisi boyutu: {features_with_mbti.shape}")
    print(f"Benzersiz MBTI tipleri: {np.unique(labels_with_mbti)}")

    # İşlenmiş özellikleri yükleme örneği
    print("\n" + "--- Kaydedilmiş Özellikleri Yükleme ---" + "\n")
    try:
        # En son kaydedilen "mbti_features_without_types" türündeki özellikleri yükler
        print("\n'without_types' özelliklerini yüklemeye çalışılıyor...")
        loaded_features_no_mbti, loaded_labels_no_mbti, loaded_feature_names_no_mbti = (
            MBTIFeatureExtractor(current_config).load_features(
                data_type="mbti_features_without_types"
            )
        )
        print(
            f"Yüklenen 'without_types' özellik matrisi boyutu: {loaded_features_no_mbti.shape}"
        )
        print(
            f"Yüklenen 'without_types' benzersiz MBTI tipleri: {np.unique(loaded_labels_no_mbti)}"
        )

        # En son kaydedilen "mbti_features_with_types" türündeki özellikleri yükler
        print("\n'with_types' özelliklerini yüklemeye çalışılıyor...")
        (
            loaded_features_with_mbti,
            loaded_labels_with_mbti,
            loaded_feature_names_with_mbti,
        ) = MBTIFeatureExtractor(current_config).load_features(
            data_type="mbti_features_with_types"
        )
        print(
            f"Yüklenen 'with_types' özellik matrisi boyutu: {loaded_features_with_mbti.shape}"
        )
        print(
            f"Yüklenen 'with_types' benzersiz MBTI tipleri: {np.unique(loaded_labels_with_mbti)}"
        )

    except FileNotFoundError as e:
        print(f"Özellikler yüklenirken hata oluştu: {e}")
    except ValueError as e:
        print(f"Özellikler yüklenirken hata oluştu: {e}")
