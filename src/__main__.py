from fastapi import FastAPI
import uvicorn

from .app import App
from .bootstrap import bootstrap
from .ioc import ioc
from .settings import load_settings


if __name__ == '__main__':
    # загрузка настроек
    settings = load_settings()
    #  создание приложения
    app = App()

    # сборка приложения
    bootstrap(app=app, settings=settings)

    try:
        # запуск приложения
        uvicorn.run(
            app=ioc.resolve(FastAPI),
            host=settings.app.host,
            port=settings.app.port,
        )
    except BaseException as e:
        print(f'{type(e)}: {e!s}.')
