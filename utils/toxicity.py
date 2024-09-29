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
    """
    Данная функция принимает на вход текст предложения, путь до места, где будет храниться модель и токенизатор, 
    название модели и порог, который определяет значение, после которого текст считается токсичным.
    """
    #Проверяет, сохранен ли уже токенайзер, и если нет, то сохраняет его локально через transformers    
    if not (Path(tokenizer_path) / 'vocab.txt').exists():
        AutoTokenizer.from_pretrained(model_name).save_pretrained(str(tokenizer_path))

    #Кладем токенизатор в переменную, считывая его по пути
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    #Проверяет, сохранен ли уже модель, и если нет, то сохраняет его локально через transformers  
    if not (Path(model_path) / 'model_sum.safetensors').exists():
        AutoModelForSequenceClassification.from_pretrained(model_name).save_pretrained(str(model_path))
    
    #Кладет модель в переменную, считывая его по пути
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    #Проверяет, доступна ли cuda на устройстве, и если да, то переносим модель на нее
    if torch.cuda.is_available():
        model.cuda()

    #Через токенизатор получаем вход для модели, и получаем логиты для каждого класса модели.
    #Модель имеет 5 классов: non-toxic, insult, obscenity, threat, dangerous
    #non-toxic показывает вероятность того, что текст НЕ токсичный
    #insult вероятность того, что текст является оскорбительным
    #obscenity вероятность того, что текст является непристойным
    #dangerous вероятность того, что текст является опаснм
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(model.device)
        proba = torch.sigmoid(model(**inputs).logits).cpu().numpy()

    #Если на входе была строка берем 0-й индекс вероятностей 
    if isinstance(text, str):
        proba = proba[0]

    #Получаем вероятность того, что текст является токсичным:
    #Он вычисляется как произведение вероятности того, что текст токсичный, множенный на вероятность опасности текст
    #Если вероятность больше порога, то возвращаем, что текст токсичный - метка True, иначe False
    return (1 - proba.T[0] * (1 - proba.T[-1])) > toxicity_threshold

#Тесты для вызова текста, как main.
if __name__ == '__main__':
    #Возвращает True - текст токсичный
    print(predict_toxicity('Шел бы ты к другому боту', './cache/toxicity/', './cache/toxicity/'))
    #Возвращает False - текст нетоксичный
    print(predict_toxicity('Вам следует сначала прочитать документацию', './cache/toxicity/', './cache/toxicity/'))
