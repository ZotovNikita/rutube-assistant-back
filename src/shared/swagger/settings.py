from typing import Optional

from pydantic import DirectoryPath, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


__all__ = ['SwaggerSettings']


class SwaggerSettings(BaseSettings):
    """Настройки для плагина сваггера"""

    model_config = SettingsConfigDict(
        extra='ignore',
        case_sensitive=False,
    )

    files_dir_path: Optional[DirectoryPath] = None

    @field_validator('files_dir_path', mode='before')
    @classmethod
    def validate_files_dir_path(cls, v: Optional[str]) -> str:
        return v or None
