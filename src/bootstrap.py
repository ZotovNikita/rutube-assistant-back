from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.shared.swagger import swagger_plugin
from src.rag import rag_plugin
from .app import App
from .ioc import ioc
from .settings import Settings


__all__ = ['bootstrap']


def bootstrap(
    app: App,
    settings: Settings,
) -> None:
    ioc.register(Settings, instance=settings)

    app.add_plugin(swagger_plugin(settings.swagger))
    app.add_plugin(rag_plugin(settings))

    @asynccontextmanager
    async def lifespan(_):
        await app.startapp()
        yield
        await app.shutdown()

    tags = [
        {'name': 'QA', 'description': 'Вопросно-Ответная система'},
        {'name': 'classification', 'description': 'Классификация'},
    ]

    fastapi = FastAPI(
        title=settings.app.title,
        lifespan=lifespan,
        version='0.1.0',
        # openapi_url='/openapi.json',
        openapi_tags=tags,
        redoc_url=None,
        **dict(docs_url=None) if settings.swagger.files_dir_path else dict(),
    )

    fastapi.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*'],
    )

    ioc.register(FastAPI, instance=fastapi)
