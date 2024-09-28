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


class BaseConfig(BaseSettings):
    model_config = SettingsConfigDict(
        extra='ignore',
        case_sensitive=False,
    )


class AppSettings(BaseConfig):
    title: str
    host: str
    port: int


class Settings(BaseConfig):
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        env_nested_delimiter='__',
    )

    app: AppSettings
    swagger: SwaggerSettings = SwaggerSettings()


def load_settings() -> Settings:
    return Settings()
