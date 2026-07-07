Проект сравнивает два подхода к выявлению языка вражды (hate speech) в русскоязычных текстах:

Классический ML: TF‑IDF + логистическая регрессия (baseline)
Современный NLP: дообученный RuBERT (DeepPavlov/rubert-base-cased)
Используется иерархическая двухуровневая классификация:

Бинарный классификатор: hate vs no‑hate
Мультиклассовый классификатор: ethnic, religious, sexism, ableism
📊 Данные

Источники: открытые датасеты + сбор из ВКонтакте и Telegram (новостные паблики)
Объём: 8 694 размеченных текста
Разметка: два уровня (бинарный + подкатегории)
Согласие аннотаторов: Cohen’s κ = 0.71 (уровень 1) и 0.83 (уровень 2)
Категория	Количество	Доля
Hate speech	3 916	45%
No hate speech	4 778	55%
🧹 Предобработка

Для baseline: очистка, приведение к нижнему регистру, токенизация (NLTK), лемматизация (pymorphy3), удаление стоп-слов, TF‑IDF (униграммы + биграммы)
Для RuBERT: минимальная очистка; токенизация и стоп-слова не удаляются (механизм внимания)
🤖 Модели

Базовый классификатор

Logistic Regression с регуляризацией C=0.05, class_weight='balanced'
Быстрый, интерпретируемый
RuBERT

DeepPavlov/rubert-base-cased (~110 млн параметров)
Fine‑tuning: 5 эпох, lr=2e‑5, batch=8, max_length=128
📈 Результаты

Уровень 1 (бинарный)

Модель	Precision (hate)	Recall (hate)	F1 (hate)	Accuracy
Baseline	0.77	0.60	0.67	74%
RuBERT	0.83	0.90	0.86	87%
Уровень 2 (мультикласс) – средний F1

Модель	Ethnic	Religious	Sexism	Ableism	Average
Baseline	0.67	0.71	0.71	0.71	0.70
RuBERT	0.91	0.91	0.90	0.96	0.92
Итоговая иерархическая точность

Baseline: 69%
RuBERT: 84% ✅
⚡ Быстрый старт

bash
git clone https://github.com/yourusername/hate-speech-detection-rubert.git
cd hate-speech-detection-rubert
pip install -r requirements.txt
jupyter notebook "RuBERT comparison.ipynb"
Важно: датасет не включён в репозиторий. Для воспроизведения подготовьте свой корпус согласно гайдлайнам (см. Приложение в дипломе).
