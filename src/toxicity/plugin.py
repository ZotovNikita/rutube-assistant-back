import re
from typing import AsyncGenerator
from itertools import chain

from src.ioc import ioc
from src.settings import Settings


__all__ = ['toxicity_plugin']


async def toxicity_plugin(settings: Settings) -> AsyncGenerator:
    """
    Данный плагин регистрирует зависимость - регулярное выражение, которое в дальнейшем
    позволяет заменить все нецензурные и оскорбительные слова на ***
    """

    # Получаем из генеральных настроек путь до оскорбительных слов
    with open(settings.toxicity_abusive_path, mode='r', encoding='utf-8') as f:
        abusive_words = f.read().split('\n')

    # Получаем из генеральных настроек путь до нецензурных слов
    with open(settings.toxicity_curse_path, mode='r', encoding='utf-8') as f:
        curse_words = f.read().split('\n')

    # Соединяем нецензурные и оскорбительные слова в единый список,
    # а затем создаем большое регулярное выражение, которое позволит их найти в предложении.
    ban_words = chain(abusive_words, curse_words)
    toxicity_pattern = re.compile(r'\b(' + '|'.join(re.escape(word) for word in ban_words) + r')\b', re.IGNORECASE)

    # Регистрируем паттерн в Айоке.
    ioc.register('toxicity_pattern', instance=toxicity_pattern)

    yield  # код выше выполнится при старте приложения, код ниже - при остановке
