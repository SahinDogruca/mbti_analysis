import pandas as pd
import numpy as np
import re
import pickle
import joblib
from typing import Dict, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import (
    XGBClassifier,
)  # Only needed for type hinting or if you re-initialize

# --- Import from config.py ---
# Assuming config.py is in the same directory or accessible via PYTHONPATH
from config import get_model_save_path, get_tfidf_vectorizer_load_path, get_model_params


# --- 1. Gerekli Fonksiyonları ve Sınıfları Tanımla/Yükle ---


# Metin temizleme fonksiyonu
def clean_text_for_prediction(text: str) -> str:
    """Metni temizle ve normalize et."""
    if not isinstance(text, str):
        return ""
    text = re.sub(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
        "",
        text,
    )
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s\.\!\?\,\;\:\-\(\)]", " ", text)
    return text.strip()


# Linguistic features çıkarma fonksiyonu
def extract_linguistic_features_for_prediction(text: str) -> Dict:
    """Metinden linguistic features çıkar."""
    if not text:
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
    personal_pronoun_count = sum(text.lower().count(word) for word in personal_pronouns)

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

    uncertainty_words = ["maybe", "perhaps", "might", "could", "possibly", "probably"]
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
        "exclamation_ratio": (exclamation_count / word_count if word_count > 0 else 0),
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
        "certainty_word_ratio": (certainty_count / word_count if word_count > 0 else 0),
        "uncertainty_word_ratio": (
            uncertainty_count / word_count if word_count > 0 else 0
        ),
        "analytical_word_ratio": (
            analytical_count / word_count if word_count > 0 else 0
        ),
    }


# --- 2. Kaydedilmiş Modeli ve Vektörleştiriciyi Yükle ---
# Modelin kaydedildiği dosya yollarını config'den alın
MODEL_PATH = get_model_save_path("xgboost_multiclass")
TFIDF_VECTORIZER_PATH = get_tfidf_vectorizer_load_path("xgboost_multiclass")


loaded_tfidf_vectorizer = None
loaded_model = None
loaded_scaler = None
loaded_label_encoder = None
loaded_feature_names = []
loaded_mbti_types = []

if TFIDF_VECTORIZER_PATH:
    try:
        with open(TFIDF_VECTORIZER_PATH, "rb") as f:
            loaded_tfidf_vectorizer = pickle.load(f)
        print(f"TF-IDF vektörleştirici başarıyla yüklendi: {TFIDF_VECTORIZER_PATH}")
    except FileNotFoundError:
        print(f"Hata: TF-IDF vektörleştirici bulunamadı: {TFIDF_VECTORIZER_PATH}")
        print(
            "Lütfen config.yaml ve model kaydetme sürecinin doğru çalıştığından emin olun."
        )
    except Exception as e:
        print(f"TF-IDF vektörleştirici yüklenirken bir hata oluştu: {e}")
else:
    print("TF-IDF vektörleştirici yolu config'den alınamadı.")


if MODEL_PATH:
    try:
        model_data = joblib.load(MODEL_PATH)
        loaded_model = model_data["model"]
        loaded_scaler = model_data["scaler"]
        loaded_label_encoder = model_data["label_encoder"]
        loaded_feature_names = model_data["feature_names"]
        loaded_mbti_types = model_data["mbti_types"]
        print(f"XGBoost modeli ve yardımcıları başarıyla yüklendi: {MODEL_PATH}")
    except FileNotFoundError:
        print(f"Hata: Model dosyası bulunamadı: {MODEL_PATH}")
        print(
            "Lütfen config.yaml ve model kaydetme sürecinin doğru çalıştığından emin olun."
        )
    except KeyError as e:
        print(f"Hata: Yüklenen model dosyasında eksik anahtar var: {e}")
        print("Model dosyasının doğru formatta kaydedildiğinden emin olun.")
    except Exception as e:
        print(f"XGBoost modeli yüklenirken bir hata oluştu: {e}")
else:
    print("Model yolu config'den alınamadı.")


# --- 3. Tahmin Fonksiyonu Tanımla ---
def predict_mbti_from_text(text: str) -> Tuple[str, float]:
    """
    Verilen metinden MBTI kişiliğini tahmin eder.

    Args:
        text (str): Tahmin edilecek kişinin yazılı metni.

    Returns:
        Tuple[str, float]: Tahmin edilen MBTI tipi ve bu tahmine olan güven seviyesi.
                            Model veya vektörleştirici yüklenemezse ('Hata', 0.0) döner.
    """
    if loaded_model is None or loaded_tfidf_vectorizer is None:
        print(
            "Tahmin için model veya TF-IDF vektörleştirici yüklenemedi. Lütfen dosyaların varlığını ve bütünlüğünü kontrol edin."
        )
        return "Model Yükleme Hatası", 0.0

    # 1. Metni temizle
    cleaned_text = clean_text_for_prediction(text)

    # 2. TF-IDF özelliklerini çıkar
    tfidf_features = loaded_tfidf_vectorizer.transform([cleaned_text]).toarray()

    # 3. Linguistic özelliklerini çıkar
    linguistic_features_dict = extract_linguistic_features_for_prediction(cleaned_text)
    linguistic_df = pd.DataFrame([linguistic_features_dict])

    # linguistic_feature_names_from_loaded listesini, modelin feature_names'inden
    # 'ling_' ile başlayanları filtreleyerek oluşturmalıyız.
    # Bu, DataFrame'in sütunlarının doğru sırada olmasını sağlar.
    linguistic_feature_names_from_loaded = [
        f for f in loaded_feature_names if f.startswith("ling_")
    ]

    # DataFrame'in sütunlarını, modelin beklediği sıraya göre yeniden düzenle
    # Eğer eksik bir linguistic özellik varsa 0 ile doldur
    for feature_name in linguistic_feature_names_from_loaded:
        if feature_name not in linguistic_df.columns:
            linguistic_df[feature_name] = 0.0  # Varsayılan olarak 0.0

    linguistic_array = linguistic_df[linguistic_feature_names_from_loaded].values
    linguistic_array = np.nan_to_num(linguistic_array)  # NaN değerleri doldur

    # 4. Tüm özellikleri birleştir
    combined_features = np.hstack([tfidf_features, linguistic_array])

    # 5. Özellikleri ölçeklendir
    # loaded_scaler'ın fit edildiği feature sayısıyla combined_features'ın feature sayısı aynı olmalı.
    # Aksi takdirde transform hatası verir.
    if combined_features.shape[1] != loaded_scaler.n_features_in_:
        print(
            f"Özellik sayısı uyuşmazlığı! Model {loaded_scaler.n_features_in_} özellik beklerken, {combined_features.shape[1]} özellik bulundu."
        )
        print(
            "Modelin eğitiminde kullanılan özellik setinin aynı olduğundan emin olun."
        )
        return "Özellik Sayısı Hatası", 0.0

    scaled_features = loaded_scaler.transform(combined_features)
    scaled_features = scaled_features.astype(np.float32)

    # 6. Tahmin yap
    predicted_encoded = loaded_model.predict(scaled_features)[0]
    probabilities = loaded_model.predict_proba(scaled_features)[0]

    # 7. Tahmini MBTI tipine geri çevir ve güven seviyesini al
    predicted_mbti_type = loaded_label_encoder.inverse_transform([predicted_encoded])[0]
    confidence = probabilities[predicted_encoded]

    return predicted_mbti_type, confidence


# --- 4. Kullanım Örneği ---
if __name__ == "__main__":
    print("\n--- MBTI Tahmin Sistemi ---")

    # Farklı metinlerle denemeler yapabilirsiniz
    text_to_predict_1 = """
    Hello there. I am Emirhan Erdem. I am a second year Physics Engineering student at Istanbul Technical University. I am from Balıkes. Apart from that, I have been interested in artificial intelligence for about a year. To improve myself in this field, I tried to learn programs such as SQL, Python, libraries in Python. At the same time, I am receiving a volunteer training on artificial intelligence from Acun Medya. Apart from that, I am constantly trying to improve myself in this field
    """

    text_to_predict_2 = """
    Hello. I am Melek Sude Günen. Maltepe University Software Engineering is in the 3rd grade. I have previously received front end training from Arı Bilgi Education Institutions in the field of eee front end. Then I started looking for an internship to improve myself in this field and I want to do my compulsory summer internship in this field. At school, I also learned languages such as Java and C. I am currently trying to develop myself in the field of Python. As one of the school projects, we had to learn SQL at school and I also did a project about SQL.
    """

    print("\n--- Metin 1 Tahmini ---")
    mbti_type_1, confidence_1 = predict_mbti_from_text(text_to_predict_1)
    print(
        f"Metin 1 için tahmin edilen MBTI tipi: {mbti_type_1} (Güven: {confidence_1:.3f})"
    )

    print("\n--- Metin 2 Tahmini ---")
    mbti_type_2, confidence_2 = predict_mbti_from_text(text_to_predict_2)
    print(
        f"Metin 2 için tahmin edilen MBTI tipi: {mbti_type_2} (Güven: {confidence_2:.3f})"
    )
