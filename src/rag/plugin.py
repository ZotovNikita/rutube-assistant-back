import re
import asyncio
from typing import AsyncGenerator
from pathlib import Path

from fastapi import FastAPI

from src.ioc import ioc
from src.settings import Settings

from pydantic import BaseModel
from joblib import load
import onnxruntime as ort
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from langchain.retrievers import EnsembleRetriever
from langchain.schema import StrOutputParser
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama.llms import OllamaLLM
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from fastapi import FastAPI
from fastapi.responses import StreamingResponse


__all__ = ['rag_plugin']


class FormatStrOutputParser(StrOutputParser):
    toxicity_pattern: re.Pattern | None = None

    def parse(self, text: str) -> str:
        if self.toxicity_pattern is not None:
            text = self.toxicity_pattern.sub('***', text)
        return text.rstrip('\n')


class QARequest(BaseModel):
    question: str


class QAResponse(BaseModel):
    answer: str
    class_1: str
    class_2: str


class ClassificationResponse(BaseModel):
    class_1: str
    class_2: str


def vectorize_texts(texts, tokenizer, embedding_model, batch_size=1):
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        if not all(isinstance(text, str) for text in batch):
            raise ValueError("All elements in the batch must be strings.")
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = embedding_model(**inputs)
        vectors.append(outputs.last_hidden_state.mean(dim=1).numpy())
    return np.vstack(vectors)



def class_predictor_factory(emb_tokenizer, emb_model, model_onnx, label_encoder):
    def predict_class(question: str) -> str:
        emb = vectorize_texts([question], emb_tokenizer, emb_model, 1)
        emb = emb.reshape(1, 1, -1)

        outputs = model_onnx.run(None, {'input': emb})
        pred = outputs[0].argmax(axis=1)

        cls = label_encoder.inverse_transform(pred)
        return cls.item()

    return predict_class



async def rag_plugin(settings: Settings) -> AsyncGenerator:
    fastapi = ioc.resolve(FastAPI)

    # QA

    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        cache_folder=settings.embedding_cache_folder,
    )

    if not (Path(settings.embedding_cache_folder) / 'vocab.txt').exists():
        AutoTokenizer.from_pretrained(settings.embedding_model).save_pretrained(str(settings.embedding_cache_folder))
    emb_tokenizer = AutoTokenizer.from_pretrained(settings.embedding_cache_folder)

    if not (Path(settings.embedding_cache_folder) / 'model_sum.safetensors').exists():
        AutoModel.from_pretrained(settings.embedding_model).save_pretrained(str(settings.embedding_cache_folder))
    emb_model = AutoModel.from_pretrained(settings.embedding_cache_folder)

    qa_db = FAISS.load_local(
        folder_path=settings.db_cache_folder,
        embeddings=embeddings,
        index_name=settings.db_index_name,
        allow_dangerous_deserialization=True,
    )

    retriever = qa_db.as_retriever(
        search_type='similarity',
        search_kwargs={
            'k': 3,
        }
    )

    bm25_retriever: BM25Retriever = load(settings.bm25_retriever_model_path)

    rag_ensemble_retriever = EnsembleRetriever(
        retrievers=[retriever, bm25_retriever],
        weights=[0.9, 0.1],
    )

    rag_template = """
Ты интеллектуальный помощник компании RUTUBE и ты очень точно отвечаешь на вопросы. Будь вежливым.
Ответь на вопрос, выбрав фрагмент из Базы Знаний (далее - БЗ), не меняя его по возможности, сохрани все имена, аббревиатуры, даты и ссылки.
Вот База Знаний:

{context}

Вопрос: {question}

Если не можешь найти ответ в Базе Знаний, вежливо скажи, что не знаешь ответ.
"""
    rag_prompt = ChatPromptTemplate.from_template(rag_template)

    model = OllamaLLM(
        model=settings.llm_model, 
        base_url=settings.llm_service_url,
    )

    def format_docs(docs):
        return '\n\n'.join([d.page_content for d in docs])

    toxicity_pattern = ioc.resolve('toxicity_pattern')

    rag_chain = (
        {'context': rag_ensemble_retriever | format_docs, 'question': RunnablePassthrough()}
        | rag_prompt
        | model
        | FormatStrOutputParser(toxicity_pattern=toxicity_pattern)
    )

    # Classification

    # emb_model = embeddings.client

    # model_cls1: KNeighborsClassifier = load('./cache/classification/model_class1.pkl')
    # le1: LabelEncoder = load('./cache/classification/le_class1.pkl')
    model_cls1 = ort.InferenceSession('./cache/classification/best_model_c1.onnx', providers=['CPUExecutionProvider'])
    le1: LabelEncoder = load('./cache/classification/label_encoder_c1.pkl')

    # model_cls2: KNeighborsClassifier = load('./cache/classification/model_class2.pkl')
    # le2: LabelEncoder = load('./cache/classification/le_class2.pkl')
    model_cls2 = ort.InferenceSession('./cache/classification/best_model_c22.onnx', providers=['CPUExecutionProvider'])
    le2: LabelEncoder = load('./cache/classification/label_encoder_c2.pkl')

    # predict_class_1 = class_predictor_factory(emb_model, model_cls1, le1, onnx=True)
    # predict_class_2 = class_predictor_factory(embeddings.client, model_cls2, le2, onnx=False)
    predict_class_1 = class_predictor_factory(emb_tokenizer, emb_model, model_cls1, le1)
    predict_class_2 = class_predictor_factory(emb_tokenizer, emb_model, model_cls2, le2)

    # similar docs

    part1_db = FAISS.load_local(
        folder_path='./cache/db/user',
        embeddings=embeddings,
        index_name='faiss.db',
        allow_dangerous_deserialization=True,
    )
    part2_db = FAISS.load_local(
        folder_path='./cache/db/content',
        embeddings=embeddings,
        index_name='faiss.db',
        allow_dangerous_deserialization=True,
    )
    part1_db.merge_from(part2_db)
    parts_retriever = part1_db.as_retriever(
        search_type='similarity',
        search_kwargs={
            'k': 3,
        }
    )

    # Views

    @fastapi.post(
        '/qa/stream',
        tags=['QA'],
        name='Получить потоковый ответ на вопрос',
        description='Получение ответа на вопрос по Базе Знаний в потоковом режиме: отправка происходит каждый сгенерированный токен.',
    )
    async def stream_answer_qa(request: QARequest) -> StreamingResponse:
        return StreamingResponse(rag_chain.astream(request.question), media_type='text/event-stream')

    @fastapi.post(
        '/qa',
        tags=['QA'],
        name='Получить ответ на вопрос и классификаторы.',
        description='Получение ответа на вопрос по Базе Знаний.\n\nОтвет также содержит классификаторы 1-го и 2-го уровней.',
    )
    async def answer_qa(request: QARequest) -> QAResponse:
        loop = asyncio.get_running_loop()

        class_1 = await loop.run_in_executor(None, predict_class_1, request.question)
        class_2 = await loop.run_in_executor(None, predict_class_2, request.question)

        answer = await rag_chain.ainvoke(request.question)

        return QAResponse(
            answer=answer,
            class_1=class_1,
            class_2=class_2,
        )

    @fastapi.post(
        '/classification',
        tags=['classification'],
        name='Классифицировать вопрос',
        description='Предсказать классификаторы 1-го и 2-го уровней.',
    )
    async def answer_classification(request: QARequest) -> ClassificationResponse:
        loop = asyncio.get_running_loop()

        class_1 = await loop.run_in_executor(None, predict_class_1, request.question)
        class_2 = await loop.run_in_executor(None, predict_class_2, request.question)

        return ClassificationResponse(
            class_1=class_1,
            class_2=class_2,
        )

    @fastapi.post(
        '/similar',
        tags=['QA'],
        name='Найти схожие фрагменты из документов',
        description='',
    )
    async def find_similar_docs(request: QARequest) -> list[Document]:
        docs = await parts_retriever.ainvoke(request.question)
        return docs


    yield
