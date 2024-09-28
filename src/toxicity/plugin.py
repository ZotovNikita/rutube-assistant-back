import re
from typing import AsyncGenerator
from itertools import chain

from src.ioc import ioc
from src.settings import Settings


__all__ = ['toxicity_plugin']


async def toxicity_plugin(settings: Settings) -> AsyncGenerator:
    with open(settings.toxicity_abusive_path, mode='r', encoding='utf-8') as f:
        abusive_words = f.read().split('\n')

    with open(settings.toxicity_curse_path, mode='r', encoding='utf-8') as f:
        curse_words = f.read().split('\n')

    ban_words = chain(abusive_words, curse_words)
    toxicity_pattern = re.compile(r'\b(' + '|'.join(re.escape(word) for word in ban_words) + r')\b', re.IGNORECASE)

    ioc.register('toxicity_pattern', instance=toxicity_pattern)

    yield
