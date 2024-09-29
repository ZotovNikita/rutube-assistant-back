from pathlib import Path
from typing import Union

from fastapi import FastAPI
from fastapi.openapi.docs import (
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)
from fastapi.responses import HTMLResponse


__all__ = ['SwaggerView']


class SwaggerView:
    """
    Представление с необходимыми функциями для локального сваггера
    """
    __slots__ = (
        '_server',
        '_swagger_sub_app_name',
    )

    def __init__(self, server: FastAPI, swagger_dir: Union[str, Path]) -> None:
        self._server = server
        self._swagger_sub_app_name = Path(swagger_dir)

    async def custom_swagger_ui_html(self) -> HTMLResponse:
        swagger_js_url = self._swagger_sub_app_name / 'swagger-ui-bundle.js'
        swagger_css_url = self._swagger_sub_app_name / 'swagger-ui.css'

        return get_swagger_ui_html(
            openapi_url=self._server.openapi_url,
            title=self._server.title,
            oauth2_redirect_url=self._server.swagger_ui_oauth2_redirect_url,
            swagger_js_url=str(swagger_js_url),
            swagger_css_url=str(swagger_css_url),
        )

    async def swagger_ui_redirect(self) -> HTMLResponse:
        return get_swagger_ui_oauth2_redirect_html()
