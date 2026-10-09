# api.md — API сервисов

> Контракты между фронтендом, сервисом-агрегатором и модельным сервисом.
> ML-логика живёт в `ml/` и здесь не описывается — см. [`ML.md`](../ML.md).

---

## 1. Схема

```
Frontend ──HTTP/JSON──▶ Aggregator (backend/, :8000) ──HTTP/JSON──▶ Model service (:8001)
                         · валидация входа                          · ml.pipeline.Pipeline.analyze()
                         · сессии диалога                           · предобработка, исправления
                                   │                                · без состояния
                                   ▼
                              Redis (:6379)
```

**Агрегатор** — единственная точка входа для фронтенда. Проверяет вход, пересылает текст и ответы
пользователя в модельный сервис и хранит сессию уточняющего диалога. Текст не изменяет:
нормализацию и исправления делает сам пайплайн.

**Модельный сервис** не хранит состояние: в каждый вызов передаётся весь текст и все ответы
пользователя, данные к этому моменту. Агрегатор хранит их в сессии, поэтому фронтенд присылает
только очередной ответ.

---

## 2. Архитектура агрегатора

```
backend/aggregator/
  handlers/       HTTP-обработчики, схемы запросов/ответов, единый формат ошибок
  services/       сценарий диалога: проверка текста → модель → сессия; проверка ответа
  repositories/   сессии (Redis) и шлюз к модельному сервису (httpx)
  domain.py       сущности (Session, Answer, AnalysisResult) и доменные ошибки
  config.py       настройки из переменных окружения
  main.py         сборка приложения: все HTTP-пути (маршруты API и путь модели) задаются в create_app()
```

Зависимости направлены только вниз: `handlers → services → repositories → domain`.

- **Сессии хранятся в Redis** (`RedisSessionRepository`): одна JSON-строка под ключом
  `session:<id>`. TTL — `SESSION_TTL_S`, продлевается при каждом чтении и записи; неактивные
  сессии удаляются сами. Агрегатор при старте проверяет Redis (`PING`) и не запускается без него.

---

## 3. API агрегатора (для фронтенда)

Базовый URL: `http://localhost:8000`. Интерактивная документация: `/docs`, схема: `/openapi.json`.

### Сценарий

```
POST /api/v1/analyze {text}
   │
   ▼
next_step != null ──▶ показать вопрос и options ──▶ POST /api/v1/sessions/{id}/answers {value}
   │                                                                  │
   │◀─────────────────────────────────────────────────────────────────┘
   ▼
next_step == null, ktru != null  → итоговый код
next_step == null, ktru == null  → товар не найден в КТРУ (см. errors), вопросов нет
```

### `POST /api/v1/analyze`

Создаёт сессию по описанию товара.

Запрос:

```json
{"text": "стл офисный на колесиках, кожзам, высота 110 см"}
```

| Поле | Тип | Ограничения |
|---|---|---|
| `text` | string | не пустой, не только пробелы, не длиннее `MAX_TEXT_LENGTH` (по умолчанию 2000) |

Ответ `200` (одинаковый для обоих эндпоинтов):

```json
{
  "session_id": "41ac195b-cafb-4204-9200-ed98db7c46f5",
  "corrections": [
    {"from": "стл", "to": "стул", "reason": "опечатка (отличается на 1 симв.)"},
    {"from": "кожзам", "to": "искусственная кожа", "reason": "приведено к значению КТРУ («Материал обивки»)"}
  ],
  "characteristics": [
    {"name": "Колесные опоры", "value": "Да", "source": "на колесиках"},
    {"name": "Материал обивки", "value": "Искусственная кожа", "source": "кожзам"},
    {"name": "Высота", "value": "1000–1200", "unit": "Миллиметр", "source": "110 см"}
  ],
  "candidates": [
    {"code": "31.01.11.150-00000003", "name": "Стул офисный", "score": 0.93},
    {"code": "31.01.11.150-00000007", "name": "Кресло офисное", "score": 0.88}
  ],
  "errors": [],
  "next_step": {
    "message": "Подходит 2 кода КТРУ. Укажите характеристику:",
    "characteristic": "Наличие подлокотников",
    "options": ["Да", "Нет"]
  },
  "ktru": null,
  "ktru_name": null,
  "steps": [
    {"stage": "Поиск по описанию", "count": 23},
    {"stage": "Высота = 1000–1200 Миллиметр", "count": 2}
  ]
}
```

| Поле | Описание |
|---|---|
| `session_id` | идентификатор сессии для ответа на вопрос |
| `corrections[]` | `from` → `to`, `reason`: опечатки, раскладка, приведение слов к значениям КТРУ |
| `characteristics[]` | `name`, `value`, `unit` (**только у числовых**, иначе поля нет), `source` — фрагмент текста пользователя, из которого извлечено значение |
| `candidates[]` | оставшиеся коды КТРУ: `code`, `name`, `score` (не больше 20) |
| `errors[]` | `type`, `message`; если есть — `characteristic`, `suggestion`. Типы: `nonexistent_value`, `out_of_range`, `unit_mismatch`, `conflict`, `product_not_found` |
| `next_step` | вопрос пользователю или `null`. `characteristic` — название характеристики; `null`, когда спрашиваем, какой товар или какой код выбрать (тогда `options` — названия товаров или коды) |
| `ktru`, `ktru_name` | итоговый код и наименование, когда код определён однозначно; иначе `null` |
| `steps[]` | как сужался список кодов: `stage` — шаг, `count` — сколько кодов осталось после него |

Агрегатор сокращает полный ответ модели (`ML.md`) до этих полей.

### `POST /api/v1/sessions/{session_id}/answers`

Ответ на текущий вопрос `next_step`. Фронтенд присылает только выбранный вариант; тип вопроса
и характеристику агрегатор берёт из сохранённого `next_step`, добавляет ответ к сессии, заново
вызывает модель с исходным текстом и всеми ответами и возвращает обновлённый результат (формат как выше).

```json
{"value": "Нет"}
```

| Поле | Правило |
|---|---|
| `value` | один из `next_step.options`, строка в точности |

### `GET /health`

`200 {"status": "ok"}` — процесс агрегатора жив. Доступность модели не проверяет.

### Ошибки

Единый формат для всех ошибок:

```json
{"error": {"code": "invalid_answer", "message": "Value 'Может' is not one of the offered options"}}
```

| HTTP | `code` | Когда |
|---|---|---|
| 404 | `session_not_found` | сессии нет (неверный id или истёк TTL) — начать заново через `/analyze` |
| 404 | `not_found` | несуществующий маршрут |
| 409 | `no_pending_question` | в сессии нет вопроса (`next_step` = `null`) |
| 422 | `validation_error` | тело запроса не соответствует схеме; подробности в `error.details` |
| 422 | `invalid_text` | текст пустой или слишком длинный |
| 422 | `invalid_answer` | `value` не входит в `next_step.options` |
| 502 | `model_bad_response` | модель вернула не-200 или ответ не в ожидаемом формате |
| 503 | `model_unavailable` | модельный сервис недоступен |
| 503 | `storage_unavailable` | Redis недоступен |
| 504 | `model_timeout` | модель не ответила за `MODEL_TIMEOUT_S` |

---

## 4. Контракт модельного сервиса (агрегатор → модель)

Агрегатор вызывает один эндпоинт. Это тонкая HTTP-обёртка над `Pipeline.analyze()` из
`ml/pipeline.py`: тело запроса передаётся в `analyze()` как есть, результат возвращается как есть.

> **Статус:** HTTP-обёртки модельного сервиса в репозитории пока нет — `Pipeline` вызывается
> только из Python. Агрегатор проверен против фейкового сервера с ответами в формате `ML.md`.

### `POST /analyze`

Запрос:

```json
{
  "raw": "стл на металлическом каркасе",
  "answers": [
    {"type": "characteristic", "characteristic": "Наличие подлокотников", "value": "Нет"}
  ]
}
```

| Поле | Описание |
|---|---|
| `raw` | текст пользователя без изменений |
| `answers` | все ответы пользователя по порядку; `[]` при первом запросе |

Ответ `200` — результат `Pipeline.analyze()` (см. [`ML.md`](../ML.md#формат-ответа)). Агрегатор
требует обязательные поля `input`, `text`; для `next_step` — `type`, `message`, `options`.
Любой другой статус или формат агрегатор превращает в `502 model_bad_response`.

---

## 5. Настройки агрегатора

| Переменная | По умолчанию | Описание |
|---|---|---|
| `MODEL_SERVICE_URL` | `http://localhost:8001` (в Docker — `http://model:8001`) | адрес модельного сервиса |
| `MODEL_TIMEOUT_S` | `30` | таймаут запроса к модели, секунды |
| `MAX_TEXT_LENGTH` | `2000` | максимальная длина текста |
| `REDIS_URL` | `redis://localhost:6379/0` (в Docker — `redis://redis:6379/0`) | адрес Redis для сессий |
| `SESSION_TTL_S` | `3600` | сколько секунд хранится неактивная сессия |
| `CORS_ORIGINS` | `*` | разрешённые origin фронтенда через запятую |

---

## 6. Запуск и проверка

```bash
cd backend
python -m venv .venv
.venv/bin/pip install -r requirements.txt          # Windows: .venv\Scripts\pip
.venv/bin/python -m pytest                         # тесты агрегатора (модель и Redis подменены)
docker run -d --name ktru-redis -p 6379:6379 redis:7-alpine
MODEL_SERVICE_URL=http://localhost:8001 REDIS_URL=redis://localhost:6379/0 \n  .venv/bin/uvicorn aggregator.main:app --port 8000
```

Docker:

```bash
docker build -t ktru-aggregator backend
docker run -p 8000:8000 -e MODEL_SERVICE_URL=http://host.docker.internal:8001 \n  -e REDIS_URL=redis://host.docker.internal:6379/0 ktru-aggregator
```

Проверка (нужны запущенные модельный сервис и Redis):

```bash
curl -s localhost:8000/health
curl -s -X POST localhost:8000/api/v1/analyze -H 'Content-Type: application/json' \
     -d '{"text": "стл на металлическом каркасе"}'
curl -s -X POST localhost:8000/api/v1/sessions/<session_id>/answers -H 'Content-Type: application/json' \
     -d '{"value": "Нет"}'
```

На Windows в Git Bash кириллица в аргументах `curl` портится — передавайте тело из UTF-8 файла:
`--data-binary @body.json`.
