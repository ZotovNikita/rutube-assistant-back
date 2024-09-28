from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# toxicity_threshold - порог после которого мы считаем сообщение токсичным
# model_path и tokenizer_path - путь до модели (первый запуск скачает модель в указанную директорию)
def predict_toxicity(
    text: str,
    tokenizer_path: str | Path,
    model_path: str | Path,
    model_name: str = 'cointegrated/rubert-tiny-toxicity',
    toxicity_threshold: int = 0.5,
):
    if not (Path(tokenizer_path) / 'vocab.txt').exists():
        AutoTokenizer.from_pretrained(model_name).save_pretrained(str(tokenizer_path))

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    if not (Path(model_path) / 'model_sum.safetensors').exists():
        AutoModelForSequenceClassification.from_pretrained(model_name).save_pretrained(str(model_path))

    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    if torch.cuda.is_available():
        model.cuda()

    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(model.device)
        proba = torch.sigmoid(model(**inputs).logits).cpu().numpy()

    if isinstance(text, str):
        proba = proba[0]

    return (1 - proba.T[0] * (1 - proba.T[-1])) > toxicity_threshold


if __name__ == '__main__':
    print(predict_toxicity('Шел бы ты к другому боту', '../cache/toxicity/', '../cache/toxicity/'))
    print(predict_toxicity('Вам следует сначала прочитать документацию', '../cache/toxicity/', '../cache/toxicity/'))
