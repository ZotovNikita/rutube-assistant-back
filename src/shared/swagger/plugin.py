from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from src.ioc import ioc
from .settings import SwaggerSettings
from .swagger_view import SwaggerView


__all__ = ['swagger_plugin']


async def swagger_plugin(settings: SwaggerSettings) -> AsyncGenerator:
    """
    Плагин для локального сваггера.
    
    Подгружаются статичные файлы и переопределяются необходимые эндпоинты.
    """
    fastapi = ioc.resolve(FastAPI)

    if settings.files_dir_path is not None:
        swagger_sub_app_name = 'swagger'

        fastapi.mount(
            f'/{swagger_sub_app_name}',
            StaticFiles(directory=settings.files_dir_path),
            name=swagger_sub_app_name,
        )

        swagger_view = SwaggerView(
            server=fastapi,
            swagger_dir=swagger_sub_app_name,
        )

        fastapi.add_api_route(
            path='/docs',
            endpoint=swagger_view.custom_swagger_ui_html,
            methods=['GET'],
            include_in_schema=False,
        )

        fastapi.add_api_route(
            path=fastapi.swagger_ui_oauth2_redirect_url,
            endpoint=swagger_view.swagger_ui_redirect,
            methods=['GET'],
            include_in_schema=False,
        )

    yield
