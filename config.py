import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import yaml


class Config:
    """Merkezi konfigürasyon yönetimi için sınıf"""

    def __init__(self, config_file: str = "config.yaml"):
        self.config_file = config_file
        self.base_dir = Path(__file__).parent
        self.config = self._load_config()
        self._ensure_directories()

    def _load_config(self) -> Dict[str, Any]:
        """Konfigürasyon dosyasını yükle"""
        config_path = self.base_dir / self.config_file

        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        else:
            # Varsayılan konfigürasyon
            default_config = self._get_default_config()
            self._save_config(default_config)
            return default_config

    def _get_default_config(self) -> Dict[str, Any]:
        """Varsayılan konfigürasyon ayarları (Sadece XGBoost ile)"""
        return {
            "paths": {
                "data_dir": "data",
                "cached_data_dir": "cached/data",
                "cached_model_dir": "cached/model",
                "cached_plots_dir": "cached/plots",  # Yeni eklendi
                "raw_data_file": "data/mbti_1.csv",
            },
            "models": {
                "xgboost_multiclass": {
                    "name": "xgboost_multiclass",
                    "file_prefix": "xgb_multi",
                    "params": {
                        "n_estimators": 100,
                        "max_depth": 6,
                        "learning_rate": 0.1,
                        "random_state": 42,
                    },
                },
            },
            "embeddings": {
                "method": "tfidf",
                "max_features": 5000,
                "min_df": 0.01,
                "max_df": 0.95,
                "ngram_range": [1, 3],
                "use_idf": True,
                "sublinear_tf": True,
            },
            "data_processing": {
                "test_size": 0.2,
                "random_state": 42,
                "validation_size": 0.1,
                "text_column": "posts",
                "target_column": "type",
                "clean_text": True,
                "remove_urls": True,
                "remove_mentions": True,
                "lowercase": True,
                "tuning_n_iter": 30,
                "tuning_cv_folds": 3,
                "early_stopping_rounds": 50,
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": "logs/training.log",
            },
            "features": {
                "use_tfidf": True,
                "use_word_count": True,
                "use_char_count": True,
                "use_sentence_count": True,
                "use_avg_word_length": True,
            },
        }

    def _save_config(self, config: Dict[str, Any]):
        """Konfigürasyonu dosyaya kaydet"""
        with open(self.base_dir / self.config_file, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    def _ensure_directories(self):
        """Gerekli dizinleri oluştur"""
        dirs_to_create = [
            self.get_path("cached_data_dir"),
            self.get_path("cached_model_dir"),
            self.get_path("cached_plots_dir"),  # Yeni eklendi
            self.get_path("data_dir"),
            self.base_dir / "logs",
        ]

        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)

    def get_path(self, key: str) -> Path:
        """Yol bilgisini al"""
        path_str = self.config["paths"][key]
        return self.base_dir / path_str

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Model konfigürasyonunu al"""
        if model_name == "xgboost_multiclass":
            return self.config["models"].get(model_name, {})
        return {}

    def get_model_path(self, model_name: str, version: Optional[str] = None) -> Path:
        """Model dosya yolunu oluştur"""
        model_config = self.get_model_config(model_name)
        prefix = model_config.get("file_prefix", model_name)

        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = f"{prefix}_{version}.pkl"
        return self.get_path("cached_model_dir") / filename

    def get_processed_data_path(
        self, data_type: str, version: Optional[str] = None, extension: str = ".npz"
    ) -> Path:
        """İşlenmiş veri dosya yolunu oluştur"""
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = f"{data_type}_{version}{extension}"
        return self.get_path("cached_data_dir") / filename

    def get_latest_processed_data_path(
        self, data_type: str, extension: str = ".npz"
    ) -> Optional[Path]:
        """En son işlenmiş veri dosyasını bul"""
        data_dir = self.get_path("cached_data_dir")
        pattern = f"{data_type}_*{extension}"
        matching_files = list(data_dir.glob(pattern))
        if not matching_files:
            return None
        return max(matching_files, key=lambda x: x.stat().st_mtime)

    def get_embeddings_config(self) -> Dict[str, Any]:
        """Embeddings konfigürasyonunu al"""
        return self.config["embeddings"]

    def get_data_processing_config(self) -> Dict[str, Any]:
        """Veri işleme konfigürasyonunu al"""
        return self.config["data_processing"]

    def get_plots_path(
        self, plot_name: str, version: Optional[str] = None, extension: str = ".png"
    ) -> Path:
        """Görsel dosya yolunu oluştur"""
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{plot_name}_{version}{extension}"
        return self.get_path("cached_plots_dir") / filename

    def update_config(self, key_path: str, value: Any):
        """Konfigürasyon değerini güncelle"""
        keys = key_path.split(".")
        current = self.config

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value
        self._save_config(self.config)

    def list_available_models(self) -> list:
        """Mevcut model tiplerini listele"""
        return list(self.config["models"].keys())

    def get_latest_model_path(self, model_name: str) -> Optional[Path]:
        """En son eğitilen model dosyasını bul"""
        model_config = self.get_model_config(model_name)
        prefix = model_config.get("file_prefix", model_name)

        model_dir = self.get_path("cached_model_dir")
        pattern = f"{prefix}_*.pkl"

        matching_files = list(model_dir.glob(pattern))
        if not matching_files:
            return None

        return max(matching_files, key=lambda x: x.stat().st_mtime)


config = Config()


def get_config():
    """Global config instance'ını al"""
    return config


def get_model_save_path(model_name: str, version: Optional[str] = None) -> Path:
    """Model kaydetme yolunu al"""
    return config.get_model_path(model_name, version)


def get_data_save_path(data_type: str, version: Optional[str] = None) -> Path:
    """Veri kaydetme yolunu al"""
    return config.get_processed_data_path(data_type, version)


def get_raw_data_path() -> Path:
    """Ham veri dosya yolunu al"""
    return config.get_path("raw_data_file")


def get_model_params(model_name: str) -> Dict[str, Any]:
    """Model parametrelerini al"""
    return config.get_model_config(model_name).get("params", {})


def get_plot_save_path(
    plot_name: str, version: Optional[str] = None, extension: str = ".png"
) -> Path:
    """Görsel kaydetme yolunu al"""
    return config.get_plots_path(plot_name, version, extension)


if __name__ == "__main__":
    print("Config sistemi test ediliyor...")
    print(f"Modeller: {config.list_available_models()}")
    print(f"XGBoost Multiclass parametreleri: {get_model_params('xgboost_multiclass')}")
    print(f"Model kaydetme yolu: {get_model_save_path('xgboost_multiclass')}")
    print(f"Veri kaydetme yolu: {get_data_save_path('mbti_features')}")
    print(f"Ham veri yolu: {get_raw_data_path()}")
    print(
        f"Görsel kaydetme yolu: {get_plot_save_path('confusion_matrix')}"
    )  # Yeni test
