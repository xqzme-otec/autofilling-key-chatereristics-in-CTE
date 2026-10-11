import { useRef, useState } from "react"
import type { BatchResult, Correction } from "./api"
import { examples } from "./data"
import { Button, Icon } from "./ui"

export function SearchForm({
  description,
  onChange,
  onSubmit,
  onUpload,
  loading,
  uploading,
}: {
  description: string
  onChange: (value: string) => void
  onSubmit: () => void
  onUpload: (file: File) => void
  loading: boolean
  uploading: boolean
}) {
  const fileInput = useRef<HTMLInputElement>(null)
  return (
    <form
      className="search-form"
      onSubmit={(event) => {
        event.preventDefault()
        onSubmit()
      }}
      role="search"
    >
      <label className="sr-only" htmlFor="description">
        Описание товара или закупочной потребности
      </label>
      <div className="search-form__row">
        <Icon name="search" size={22} />
        <input
          autoComplete="off"
          id="description"
          maxLength={2000}
          onChange={(event) => onChange(event.target.value)}
          placeholder="Например: пальто для девочки, молоток слесарный или офисный стол"
          type="search"
          value={description}
        />
        <Button
          disabled={!description.trim() || uploading}
          loading={loading}
          type="submit"
          variant="primary"
        >
          {loading ? "Ищем" : "Найти"}
        </Button>
      </div>
      <div className="search-form__extras">
        <div className="search-form__examples">
          <span>Примеры:</span>
          {examples.map((item) => (
            <button
              className="chip"
              key={item.label}
              onClick={() => onChange(item.text)}
              type="button"
            >
              {item.label}
            </button>
          ))}
        </div>
        <input
          accept=".csv,.xlsx,.xls"
          className="sr-only"
          onChange={(event) => {
            const file = event.currentTarget.files?.[0]
            event.currentTarget.value = ""
            if (file) onUpload(file)
          }}
          ref={fileInput}
          tabIndex={-1}
          type="file"
        />
        <div className="search-form__upload-group">
          <button
            className="search-form__upload"
            disabled={uploading}
            onClick={() => fileInput.current?.click()}
            type="button"
          >
            <Icon name="doc" size={16} />
            {uploading ? "Обрабатываем таблицу…" : "Загрузить CSV или Excel"}
          </button>
          <small className="search-form__upload-hint">
            Размер загрузки до 1 МБ · до 50 товаров
          </small>
        </div>
      </div>
    </form>
  )
}

export function BatchResults({
  batch,
  fileName,
  selectedRow,
  onSelect,
}: {
  batch: BatchResult
  fileName: string
  selectedRow: number | null
  onSelect: (row: number) => void
}) {
  const ready = batch.items.filter((item) => item.result).length
  return (
    <section aria-label="Результаты загрузки таблицы" className="batch-results">
      <div className="batch-results__head">
        <div>
          <h2>Товары из файла</h2>
          <p>
            {fileName} · обработано {ready} из {batch.items.length}
          </p>
        </div>
      </div>
      <ul>
        {batch.items.map((item) => (
          <li
            className={selectedRow === item.row ? "is-selected" : ""}
            key={item.row}
          >
            <span className="batch-results__row">Строка {item.row}</span>
            <span className="batch-results__details">
              <strong>{item.name || item.description}</strong>
              {item.name && <small>{item.description}</small>}
              {item.error && (
                <small className="batch-results__error">
                  {item.error.message}
                </small>
              )}
            </span>
            {item.result && (
              <button
                aria-pressed={selectedRow === item.row}
                onClick={() => onSelect(item.row)}
                type="button"
              >
                {selectedRow === item.row ? "Открыто" : "Открыть результат"}
              </button>
            )}
          </li>
        ))}
      </ul>
    </section>
  )
}

export function SearchCorrections({
  corrections,
  onApply,
  disabled = false,
}: {
  corrections: Correction[]
  onApply: (correction: Correction, replacement: string) => string | null
  disabled?: boolean
}) {
  const [editingIndex, setEditingIndex] = useState<number | null>(null)
  const [replacement, setReplacement] = useState("")
  const [editError, setEditError] = useState("")
  if (corrections.length === 0) return null
  return (
    <section
      aria-label="Исправления и нормализация"
      className="search-corrections"
    >
      <h2>Исправления и нормализация</h2>
      <ul>
        {corrections.map((item, index) => (
          <li key={`${item.from}-${item.to}-${index}`}>
            <Icon name="check" size={20} />
            <span className="search-corrections__content">
              <span className="search-corrections__change">
                <s>{item.from}</s>
                <span aria-hidden="true">→</span>
                <strong>{item.to}</strong>
                <Button
                  disabled={disabled}
                  onClick={() => {
                    setEditingIndex(index)
                    setReplacement(item.to)
                    setEditError("")
                  }}
                  size="sm"
                  variant="link"
                >
                  Изменить
                </Button>
              </span>
              {item.reason && <small>{item.reason}</small>}
              {editingIndex === index && (
                <form
                  className="search-corrections__edit-form"
                  onSubmit={(event) => {
                    event.preventDefault()
                    setEditError(onApply(item, replacement.trim()) || "")
                  }}
                >
                  <label htmlFor={`correction-${index}`}>Заменить в запросе на</label>
                  <div className="search-corrections__edit-row">
                    <input
                      autoFocus
                      id={`correction-${index}`}
                      maxLength={2000}
                      onChange={(event) => {
                        setReplacement(event.target.value)
                        setEditError("")
                      }}
                      value={replacement}
                    />
                    <Button disabled={!replacement.trim() || disabled} size="sm" type="submit" variant="primary">
                      Сохранить и найти
                    </Button>
                    <Button
                      onClick={() => {
                        setEditingIndex(null)
                        setEditError("")
                      }}
                      size="sm"
                    >
                      Отмена
                    </Button>
                  </div>
                  {editError && <small className="search-corrections__edit-error" role="alert">{editError}</small>}
                </form>
              )}
            </span>
          </li>
        ))}
      </ul>
    </section>
  )
}

