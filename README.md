# rutube-assistant-back

Backend &amp; ML for RUTUBE AI assistant

## Инструкция по установке

1. Установить python v3.11.9

2. Установить torch с CUDA на вашей машине (для windows: `pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu118`)

3. Выполнить `pip install -r ./req.txt`

4. Файл `.env.example` переименовать в `.env` и заполнить необходимые переменные окружения

5. Выполнить `python -m src`

По умолчанию Swagger развернут по адресу `http://localhost:8558/docs`.

Получение ответа на вопрос и классификаторов 1-го и 2-го уровней доступно по запросу `POST http://localhost:8558/qa`.
