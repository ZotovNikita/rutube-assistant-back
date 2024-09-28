from fastapi import FastAPI
import uvicorn

from .app import App
from .bootstrap import bootstrap
from .ioc import ioc
from .settings import load_settings


if __name__ == '__main__':
    settings = load_settings()
    app = App()

    bootstrap(app=app, settings=settings)

    try:
        uvicorn.run(
            app=ioc.resolve(FastAPI),
            host=settings.app.host,
            port=settings.app.port,
        )
    except BaseException as e:
        print(f'{type(e)}: {e!s}.')
