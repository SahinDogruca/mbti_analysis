# predict.py
import numpy as np
import re
import pickle
import joblib
from typing import Dict, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import sys

# Add the parent directory to the path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import TFIDF_VECTORIZER_PATH, XGBOOST_MODEL_PATH


# Re-define necessary parts from MBTI_TFIDF_FeatureExtractor for prediction
class TextProcessor:
    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = re.sub(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            "",
            text,
        )
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"\\s+", " ", text)
        text = re.sub(r"[^\\w\\s\\.\\!\\?\\,\\;\\:\\-\\(\\)]", " ", text)
        return text.strip()

    def extract_linguistic_features(self, text: str) -> Dict:
        # This is a simplified version; in a real scenario, you might want to load
        # the linguistic feature column names from a saved file or config
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


class MBTIPredictor:
    def __init__(self):
        self.tfidf_vectorizer = None
        self.model_data = None
        self.text_processor = TextProcessor()
        self._load_assets()

    def _load_assets(self):
        # Load TF-IDF Vectorizer
        try:
            with open(TFIDF_VECTORIZER_PATH, "rb") as f:
                self.tfidf_vectorizer = pickle.load(f)
            print(f"TF-IDF vektörleştirici başarıyla yüklendi: {TFIDF_VECTORIZER_PATH}")
        except FileNotFoundError:
            print(f"Hata: TF-IDF vektörleştirici bulunamadı: {TFIDF_VECTORIZER_PATH}")
            sys.exit(1)

        # Load XGBoost Model and other components
        try:
            self.model_data = joblib.load(XGBOOST_MODEL_PATH)
            print(f"XGBoost modeli başarıyla yüklendi: {XGBOOST_MODEL_PATH}")
        except FileNotFoundError:
            print(f"Hata: XGBoost modeli bulunamadı: {XGBOOST_MODEL_PATH}")
            sys.exit(1)
        except Exception as e:
            print(f"Hata: XGBoost modeli yüklenirken bir sorun oluştu: {e}")
            sys.exit(1)

    def predict_mbti_from_text(self, text: str) -> Tuple[str, float]:
        cleaned_text = self.text_processor.clean_text(text)

        # TF-IDF Features
        if self.tfidf_vectorizer:
            tfidf_features = self.tfidf_vectorizer.transform([cleaned_text]).toarray()
        else:
            raise RuntimeError("TF-IDF Vectorizer yüklenemedi.")

        # Linguistic Features
        linguistic_features_dict = self.text_processor.extract_linguistic_features(
            cleaned_text
        )
        # Ensure the order of linguistic features matches the training
        # We need the linguistic_feature_cols from the training phase.
        # This is ideally stored in the model_data or a separate config.
        # For simplicity, assuming the order from MBTI_TFIDF_FeatureExtractor
        linguistic_feature_cols = [
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
        linguistic_features = np.array(
            [linguistic_features_dict[col] for col in linguistic_feature_cols]
        ).reshape(1, -1)
        linguistic_features = np.nan_to_num(linguistic_features)

        # Combine features
        combined_features = np.hstack([tfidf_features, linguistic_features])

        # Scale features
        scaler = self.model_data["scaler"]
        scaled_features = scaler.transform(combined_features).astype(np.float32)

        # Predict
        model = self.model_data["model"]
        label_encoder = self.model_data["label_encoder"]

        predicted_encoded = model.predict(scaled_features)[0]
        probabilities = model.predict_proba(scaled_features)[0]

        predicted_mbti_type = label_encoder.inverse_transform([predicted_encoded])[0]
        confidence = probabilities[predicted_encoded]

        return predicted_mbti_type, confidence


if __name__ == "__main__":
    predictor = MBTIPredictor()

    text_to_predict_1 = """
    "Hayatımı organize etmeyi ve planlamayı severim. Her zaman geleceği düşünürüm ve her şeyin mantıklı bir açıklaması olması gerektiğini inanırım. 
    Detaylara dikkat ederim ve kararlarımı genellikle verilere dayanarak alırım. Sosyal ortamlar beni yorsa da, derin ve anlamlı sohbetlerden keyif alırım. 
    İnsanların potansiyellerini görmelerine yardımcı olmayı severim."
    """

    text_to_predict_2 = """
    "Spontan ve uyumlu bir insanım. Yeni deneyimlere açığım ve anı yaşamayı tercih ederim. 
    Çoğu zaman duygularımla hareket ederim ve etrafımdaki insanların hislerine karşı hassasımdır. 
    Mantıktan çok sezgilere güvenirim. Sanatsal ve yaratıcı uğraşlarla ilgilenmeyi severim."
    """

    print("\n--- Metin 1 Tahmini ---")
    mbti_type_1, confidence_1 = predictor.predict_mbti_from_text(text_to_predict_1)
    print(
        f"Metin 1 için tahmin edilen MBTI tipi: {mbti_type_1} (Güven: {confidence_1:.3f})"
    )

    print("\n--- Metin 2 Tahmini ---")
    mbti_type_2, confidence_2 = predictor.predict_mbti_from_text(text_to_predict_2)
    print(
        f"Metin 2 için tahmin edilen MBTI tipi: {mbti_type_2} (Güven: {confidence_2:.3f})"
    )
