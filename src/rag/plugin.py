from typing import AsyncGenerator

from fastapi import FastAPI

from src.ioc import ioc
from src.settings import Settings

from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama.llms import OllamaLLM
from fastapi import FastAPI
from fastapi.responses import StreamingResponse


__all__ = ['rag_plugin']


class QARequest(BaseModel):
    question: str


class QAResponse(BaseModel):
    answer: str
    class_1: str
    class_2: str


class ClassificationResponse(BaseModel):
    class_1: str
    class_2: str


async def predict_cls(question: str) -> tuple[str, str]:
    return 'class_1', 'class_2'


async def rag_plugin(settings: Settings) -> AsyncGenerator:
    fastapi = ioc.resolve(FastAPI)

    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        cache_folder=settings.embedding_cache_folder,
    )

    db = FAISS.load_local(
        folder_path=settings.db_cache_folder,
        embeddings=embeddings,
        index_name=settings.db_index_name,
        allow_dangerous_deserialization=True,
    )

    retriever = db.as_retriever(
        search_type='similarity',
        search_kwargs={
            'k': 3,
        }
    )

    template = """
Ты интеллектуальный помощник компании RUTUBE и ты очень точно отвечаешь на вопросы. Будь вежливым.
Ответь на вопрос, выбрав фрагмент из Базы Знаний (далее - БЗ), не меняя его по возможности, сохрани все имена, аббревиатуры, даты и ссылки.
Вот База Знаний:

{context}

Вопрос: {question}

Если не можешь найти ответ в контексте, вежливо скажи, что не знаешь ответ
"""
    prompt = ChatPromptTemplate.from_template(template)

    model = OllamaLLM(
        model=settings.llm_model, 
        base_url=settings.llm_service_url,
    )

    def format_docs(docs):
        return '\n\n'.join([d.page_content for d in docs])

    chain = (
        {'context': retriever | format_docs, 'question': RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    @fastapi.post(
        '/qa/stream',
        tags=['qa'],
        name='Получить потоковый ответ на вопрос',
        description='Получение ответа на вопрос по Базе Знаний в потоковом режиме: отправка происходит каждый сгенерированный токен',
    )
    async def stream_answer_qa(request: QARequest) -> StreamingResponse:
        return StreamingResponse(chain.astream(request.question), media_type='text/event-stream')

    @fastapi.post(
        '/qa',
        tags=['qa'],
        name='Получить ответ на вопрос',
        description='Получение ответа на вопрос по Базе Знаний. Ответ также содержит метки классов',
    )
    async def answer_qa(request: QARequest) -> QAResponse:
        class_1, class_2 = await predict_cls(request.question)

        answer = await chain.ainvoke(request.question)

        return QAResponse(
            answer=answer,
            class_1=class_1,
            class_2=class_2,
        )

    @fastapi.post(
        '/classification',
        tags=['classification'],
        name='Классифицировать вопрос',
        description='Предсказать классификаторы 1-го и 2-го уровней',
    )
    async def answer_classification(request: QARequest) -> ClassificationResponse:
        class_1, class_2 = await predict_cls(request.question)

        return ClassificationResponse(
            class_1=class_1,
            class_2=class_2,
        )

    yield
