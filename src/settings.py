from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.shared.swagger import SwaggerSettings


__all__ = [
    'AlgSettings',
    'BaseConfig',
    'DbSettings',
    'MainSettings',
    'NNSettings',
    'Settings',
    'TgBotSettings',
    'TrackerSettings',
    'load_settings',
]


# Базовый класс настроек с дефолтными параметрами
class BaseConfig(BaseSettings):
    model_config = SettingsConfigDict(
        extra='ignore',
        case_sensitive=False,
    )


# Настройки веб части приложения
class AppSettings(BaseConfig):
    title: str
    host: str
    port: int


# Настройки всего приложения
class Settings(BaseConfig):
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        env_nested_delimiter='__',  # позволяет задать настройки вложенным сущностям (например, APP__HOST) 
    )

    embedding_cache_folder: str = './cache/embedding/'
    embedding_model: str = 'deepvk/USER-bge-m3'

    db_cache_folder: str = './cache/db/faiss'
    db_index_name: str

    bm25_retriever_model_path: str = './cache/bm25.pkl'

    llm_model: str = 'gemma2:9b'
    llm_service_url: str = 'http://localhost:11434'

    toxicity_abusive_path: Path = './data/toxicity/ru_abusive_words.txt'
    toxicity_curse_path: Path = './data/toxicity/ru_curse_words.txt'

    app: AppSettings
    swagger: SwaggerSettings = SwaggerSettings()


# Функция для создания настроек (подгрузит данные из .env)
def load_settings() -> Settings:
    return Settings()
