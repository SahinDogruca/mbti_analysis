# config.py
import os
from pathlib import Path

# Base Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")  # New directory for plots

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)  # Ensure plots directory exists

# Data Paths
MBTI_DATASET_PATH = os.path.join(DATA_DIR, "mbti_1.csv")
EMBEDDINGS_FILE_PATH = os.path.join(PROCESSED_DATA_DIR, "mbti_embeddings_tfidf")
EMBEDDINGS_FILE_PATH_BERT = os.path.join(
    PROCESSED_DATA_DIR, "mbti_embeddings_tfidf_bert"
)
TFIDF_VECTORIZER_PATH = os.path.join(
    PROCESSED_DATA_DIR, "mbti_embeddings_tfidf_vectorizer.pkl"
)

# Model Paths
XGBOOST_MODEL_PATH = os.path.join(MODEL_DIR, "mbti_xgboost_multiclass_model.pkl")
SVC_MODEL_PATH = os.path.join(MODEL_DIR, "mbti_svc_multiclass_model.pkl")

# Feature Extraction Parameters
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 3)
USE_BERT_EMBEDDINGS = True
BERT_MODEL_NAME = "bert-large-uncased"

# Training Parameters
TEST_SIZE = 0.2
VAL_SIZE = 0.1
RANDOM_STATE = 42
USE_RANDOM_SEARCH = False

# Class Imbalance Handling Parameters (New)
# Set to 'none', 'smote', or 'class_weight'
CLASS_IMBALANCE_STRATEGY = "smote"  # Varsayılan olarak dengeleme yapılmaz
SMOTE_K_NEIGHBORS = 5  # SMOTE için k_neighbors değeri

# Neural Network Parameters (New)
NEURAL_NETWORK_LEARNING_RATE = 0.001
NEURAL_NETWORK_EPOCHS = 50
NEURAL_NETWORK_BATCH_SIZE = 32
