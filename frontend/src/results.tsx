import type { AnalysisResult, Correction } from "./api"
import { Button, Card, Icon, Spinner, Status } from "./ui"

function normalizedName(name: string): string {
  return name.trim().toLocaleLowerCase("ru")
}

function correctedText(text: string, corrections: Correction[]) {
  const correction = corrections.find(
    (item) => normalizedName(item.from) === normalizedName(text),
  )
  return correction?.to || text
}

function characteristicOrigin(origin: string | null | undefined) {
  if (origin === "ktru") return "КТРУ"
  if (origin === "portal") return "Портал поставщиков"
  if (origin === "new") return "Новая характеристика"
  if (origin === "answer") return "Ответ пользователя"
  return null
}

function errorLabel(severity: string | undefined) {
  if (severity === "info") return "Информация"
  if (severity === "warning") return "Стоит проверить"
  return "Ошибка в описании"
}

function characteristicNotice(name: string, errors: AnalysisResult["errors"]) {
  return errors.find(
    (error) =>
      (error.type === "not_fitting_characteristic" ||
        error.type === "uncertain_characteristic") &&
      (error.characteristic == null
        ? undefined
        : normalizedName(error.characteristic)) === normalizedName(name),
  )
}

export function CharacteristicsCard({
  result,
  loading,
}: {
  result: AnalysisResult | null
  loading: boolean
}) {
  if (loading)
    return (
      <Card className="chars">
        <h2>Распознанные характеристики</h2>
        <div className="api-loading">
          <Spinner /> Сопоставляем описание со справочником…
        </div>
      </Card>
    )
  if (!result) return null
  const foundNames = new Set(result.characteristics.map((item) => normalizedName(item.name)))
  const missingRequired = [...new Set(result.missing_required ?? [])].filter(
    (name) => !foundNames.has(normalizedName(name)),
  )
  return (
    <Card className="chars" id="characteristics">
      <div className="card__head">
        <h2>Характеристики товара</h2>
        <span className="hint">
          Найдено: {result.characteristics.length}
          {missingRequired.length > 0 && (
            <> · Не указано: {missingRequired.length}</>
          )}
        </span>
      </div>
      {(result.product_intent || result.subtype_context?.length > 0) && (
        <section className="api-section">
          <h3>Товар и контекст</h3>
          <p>
            {result.product_intent && (
              <>
                Товар: <strong>{result.product_intent}</strong>
              </>
            )}
            {result.subtype_context?.length > 0 && (
              <>
                {" "}
                · Контекст:{" "}
                {result.subtype_context
                  .map((signal) => signal.value)
                  .join(", ")}
              </>
            )}
          </p>
        </section>
      )}
      {result.errors.length > 0 && (
        <section className="api-section" aria-label="Сообщения об описании">
          <h3>Об описании</h3>
          <ul className="api-list api-list--messages">
            {result.errors.map((item, index) => (
              <li
                className={`api-message api-message--${item.severity || "error"}`}
                key={`${item.type}-${index}`}
              >
                <Icon
                  name={item.severity === "info" ? "info" : "alert"}
                  size={16}
                />
                <span>
                  <b className="api-message__label">
                    {errorLabel(item.severity)}
                  </b>
                  <strong>{item.characteristic || "Описание товара"}</strong>:{" "}
                  {item.message}
                  {item.suggestion && (
                    <small>Предложение: {item.suggestion}</small>
                  )}
                </span>
              </li>
            ))}
          </ul>
        </section>
      )}
      {result.characteristics.length > 0 || missingRequired.length > 0 ? (
        <div
          aria-label="Характеристики товара"
          className="api-table"
          role="table"
        >
          <div className="api-table__head" role="row">
            <span role="columnheader">Характеристика</span>
            <span role="columnheader">Значение</span>
          </div>
          {result.characteristics.map((item, index) => {
            const notice = characteristicNotice(item.name, result.errors)
            const origin = characteristicOrigin(item.origin)
            return (
              <div
                className="api-table__row"
                key={`${item.name}-${index}`}
                role="row"
              >
                <span role="cell">
                  <strong>
                    {correctedText(item.name, result.corrections)}
                  </strong>
                  {origin && (
                    <small className="api-table__origin">
                      {origin}
                    </small>
                  )}
                  {item.origin === "new" && typeof item.known === "boolean" && (
                    <small className="api-table__known">
                      {item.known
                        ? "Уже встречалась у этого товара"
                        : "Ранее не встречалась у этого товара"}
                    </small>
                  )}
                  {notice && (
                    <small
                      className={`api-table__notice api-table__notice--${
                        notice.type === "not_fitting_characteristic"
                          ? "warning"
                          : "info"
                      }`}
                    >
                      {notice.type === "not_fitting_characteristic"
                        ? "Возможно, не подходит товару"
                        : "Не уверены, что относится к товару"}
                    </small>
                  )}
                </span>
                <span role="cell">
                  {correctedText(item.value, result.corrections)}
                  {item.unit ? ` ${item.unit}` : ""}
                </span>
              </div>
            )
          })}
          {missingRequired.map((name) => (
            <div
              className="api-table__row api-table__row--missing"
              key={`missing-${name}`}
              role="row"
            >
              <span role="cell">
                <strong>{name}</strong>
                <small className="api-table__origin">
                  Обязательная по КТРУ
                </small>
              </span>
              <span role="cell">Не указано</span>
            </div>
          ))}
        </div>
      ) : (
        <p className="api-empty">
          Характеристики пока не найдены. Уточните описание и повторите поиск.
        </p>
      )}
    </Card>
  )
}

function plural(count: number) {
  const mod10 = count % 10,
    mod100 = count % 100
  if (mod10 === 1 && mod100 !== 11) return "позиция"
  if (mod10 >= 2 && mod10 <= 4 && (mod100 < 12 || mod100 > 14)) return "позиции"
  return "позиций"
}

export function KtruCard({
  result,
  loading,
}: {
  result: AnalysisResult | null
  loading: boolean
}) {
  if (loading && !result)
    return (
      <Card className="ktru">
        <div className="ktru-head">
          <h2>Подбор кода КТРУ</h2>
        </div>
        <div className="api-loading">
          <Spinner /> Ищем позиции…
        </div>
      </Card>
    )
  if (!result)
    return (
      <Card className="ktru">
        <div className="ktru-head">
          <h2>Подбор кода КТРУ</h2>
          <Status tone="neutral">Ожидает данных</Status>
        </div>
        <div className="ktru-empty">
          <p>Подбор начнётся после анализа описания.</p>
        </div>
      </Card>
    )
  const complete = result.ktru !== null
  const lastCount =
    result.remaining_count ??
    result.steps.at(-1)?.count ??
    result.candidates.length
  return (
    <Card className="ktru">
      <div className="ktru-head">
        <div className="ktru-head__row">
          <h2>Подбор кода КТРУ</h2>
          <Status
            tone={complete ? "success" : result.next_step ? "info" : "warning"}
          >
            {complete
              ? "Код определён"
              : result.next_step
                ? "Нужно уточнение"
                : "Нет результата"}
          </Status>
        </div>
        {!complete && (
          <p className="ktru-head__count">
            Поиск: {result.retrieval_count ?? result.candidates.length} · По
            товару: {result.compatible_count ?? lastCount} · Осталось:{" "}
            <b>{lastCount}</b> {plural(lastCount)}
          </p>
        )}
      </div>
      {result.steps.length > 0 && (
        <ol aria-label="Сужение списка КТРУ" className="history">
          {result.steps.map((step, index) => (
            <li
              className={index === result.steps.length - 1 ? "is-last" : ""}
              key={`${step.stage}-${index}`}
            >
              <span className="history__dot" />
              <span className="history__label">{step.stage}</span>
              <span className="history__count">{step.count}</span>
            </li>
          ))}
        </ol>
      )}
      {complete && (
        <div className="api-final">
          <span className="label-caps">Позиция КТРУ</span>
          <h3>{result.ktru_name || "Товар"}</h3>
          <strong className="api-code">{result.ktru}</strong>
        </div>
      )}
      {!complete && !result.next_step && (
        <p className="api-empty">
          Однозначный код не найден. Уточните описание и запустите подбор
          заново.
        </p>
      )}
      {!complete && result.candidates.length > 0 && (
        <div className="api-candidates">
          <h3>Подходящие позиции</h3>
          <ul>
            {result.candidates.map((candidate) => (
              <li key={candidate.code}>
                <strong>{candidate.name}</strong>
                <span>{candidate.code}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </Card>
  )
}

export function ClarifyingQuestion({
  result,
  loading,
  selectedAnswer,
  onAnswer,
  onSubmitAnswer,
}: {
  result: AnalysisResult | null
  loading: boolean
  selectedAnswer: string
  onAnswer: (value: string) => void
  onSubmitAnswer: () => void
}) {
  if (!result?.next_step) return null
  return (
    <Card className="question" id="question">
      <span className="label-caps label-caps--blue">Уточняющий вопрос</span>
      <h3>{result.next_step.message}</h3>
      {result.next_step.characteristic && (
        <p>Характеристика: {result.next_step.characteristic}</p>
      )}
      <fieldset className="radios" disabled={loading}>
        <legend className="sr-only">{result.next_step.message}</legend>
        {result.next_step.options.map((option) => (
          <label
            className={`radio ${selectedAnswer === option ? "is-checked" : ""}`}
            key={option}
          >
            <input
              checked={selectedAnswer === option}
              name="clarify"
              onChange={() => onAnswer(option)}
              type="radio"
              value={option}
            />
            <span className="radio__mark" />
            {option}
          </label>
        ))}
      </fieldset>
      <Button
        className="btn--block"
        disabled={!selectedAnswer}
        loading={loading}
        onClick={onSubmitAnswer}
        variant="primary"
      >
        {loading ? "Уточняем подбор" : "Уточнить подбор"}
      </Button>
    </Card>
  )
}
