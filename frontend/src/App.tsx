import { useEffect, useRef, useState } from "react"
import {
  analyze,
  analyzeBatch,
  answerQuestion,
  ApiError,
  type AnalysisResult,
  type BatchResult,
  type Characteristic,
} from "./api"
import { NotificationBar } from "./ui"
import { Footer, Header, PageHead } from "./layout"
import { CharacteristicsCard, ClarifyingQuestion, KtruCard } from "./results"
import { BatchResults, SearchCorrections, SearchForm } from "./search"

function analysisErrorMessage(cause: unknown): string {
  if (!(cause instanceof ApiError)) {
    return "Не удалось обработать запрос. Попробуйте ещё раз."
  }
  if (cause.code === "session_not_found") {
    return "Сессия больше не доступна. Запустите подбор заново по сохранённому описанию."
  }
  if (
    ["model_unavailable", "model_timeout", "model_bad_response"].includes(
      cause.code,
    )
  ) {
    return "Модельный сервис пока не ответил. Повторите запрос позже."
  }
  return cause.message
}

function logRequestError(cause: unknown): void {
  if (cause instanceof ApiError) {
    console.error(
      `Ошибка запроса к API: code=${cause.code}, status=${cause.status}, request_id=${cause.requestId ?? "не получен"}`,
    )
    return
  }
  console.error("Ошибка запроса к API: code=unexpected_error, request_id=не получен")
}

export default function App() {
  const [description, setDescription] = useState(() => {
    try {
      return localStorage.getItem("ste-description") || ""
    } catch {
      return ""
    }
  })
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [selectedAnswer, setSelectedAnswer] = useState("")
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")
  const [batch, setBatch] = useState<BatchResult | null>(null)
  const [batchFileName, setBatchFileName] = useState("")
  const [selectedBatchRow, setSelectedBatchRow] = useState<number | null>(null)
  const [uploading, setUploading] = useState(false)
  const activeRequest = useRef<AbortController | null>(null)
  const resultSection = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    try {
      localStorage.setItem("ste-description", description)
    } catch {
      /* Storage may be unavailable. */
    }
  }, [description])
  useEffect(() => () => activeRequest.current?.abort(), [])

  const beginRequest = () => {
    activeRequest.current?.abort()
    const controller = new AbortController()
    activeRequest.current = controller
    return controller
  }

  const changeDescription = (value: string) => {
    activeRequest.current?.abort()
    activeRequest.current = null
    setLoading(false)
    setUploading(false)
    setDescription(value)
    setResult(null)
    setSelectedAnswer("")
    setError("")
  }

  const perform = async (
    request: (signal: AbortSignal) => Promise<AnalysisResult>,
  ) => {
    const controller = beginRequest()
    setLoading(true)
    setError("")
    try {
      const next = await request(controller.signal)
      if (activeRequest.current !== controller) return
      setResult(next)
      if (selectedBatchRow !== null)
        setBatch(
          (current) =>
            current && {
              items: current.items.map((item) =>
                item.row === selectedBatchRow
                  ? { ...item, result: next }
                  : item,
              ),
            },
        )
      setSelectedAnswer("")
    } catch (cause) {
      if (controller.signal.aborted || activeRequest.current !== controller)
        return
      logRequestError(cause)
      setError(analysisErrorMessage(cause))
    } finally {
      if (activeRequest.current === controller) {
        activeRequest.current = null
        setLoading(false)
      }
    }
  }

  const startAnalysis = (text = description) => {
    if (!text.trim()) return
    setBatch(null)
    setSelectedBatchRow(null)
    setResult(null)
    setSelectedAnswer("")
    void perform((signal) => analyze(text, signal))
  }
  const analyzeEditedDescription = (text: string): string | null => {
    if (!text.trim()) return "Введите описание товара."
    if (text.length > 2000) return "Описание длиннее 2000 символов. Сократите его."
    if (text === description) return "Измените описание перед повторным поиском."
    changeDescription(text)
    startAnalysis(text)
    return null
  }
  const applyCorrection = (correction: { from: string }, replacement: string): string | null => {
    const original = correction.from.trim()
    const index = description.toLocaleLowerCase("ru").indexOf(original.toLocaleLowerCase("ru"))
    if (!original || index < 0) return "Не удалось найти это слово в запросе. Измените текст в строке поиска."
    const updated = description.slice(0, index) + replacement + description.slice(index + original.length)
    if (updated.length > 2000) return "Описание длиннее 2000 символов. Сократите замену."
    changeDescription(updated)
    startAnalysis(updated)
    return null
  }
  const applyCharacteristic = (item: Characteristic | null, name: string, value: string): string | null => {
    const label = name.trim()
    const answer = value.trim()
    if (!label || !answer) return "Укажите название характеристики и её значение."
    const phrase = `${label}: ${answer}`
    let updated: string
    if (item) {
      const source = (item.user_text || (item.source && !["ktru", "portal", "user"].includes(item.source) ? item.source : "") || "").trim()
      const index = source ? description.toLocaleLowerCase("ru").indexOf(source.toLocaleLowerCase("ru")) : -1
      if (index < 0) return "Не удалось найти исходный фрагмент в запросе. Исправьте описание в строке поиска и нажмите «Найти»."
      updated = description.slice(0, index) + phrase + description.slice(index + source.length)
    } else {
      updated = `${description.trim()}${description.trim() ? ", " : ""}${phrase}`
    }
    if (updated.length > 2000) return "Описание длиннее 2000 символов. Сократите значение."
    changeDescription(updated)
    startAnalysis(updated)
    return null
  }
  const submitAnswer = () => {
    if (
      !result?.next_step ||
      !result.next_step.options.includes(selectedAnswer)
    )
      return
    void perform((signal) =>
      answerQuestion(result.session_id, selectedAnswer, signal),
    )
  }

  const uploadBatchFile = async (file: File) => {
    const controller = beginRequest()
    setLoading(false)
    setUploading(true)
    setError("")
    try {
      const next = await analyzeBatch(file, controller.signal)
      if (activeRequest.current !== controller) return
      setBatch(next)
      setBatchFileName(file.name)
      setSelectedBatchRow(null)
      setResult(null)
    } catch (cause) {
      if (controller.signal.aborted || activeRequest.current !== controller)
        return
      logRequestError(cause)
      setError(
        cause instanceof ApiError
          ? cause.message
          : "Не удалось обработать таблицу. Попробуйте ещё раз.",
      )
    } finally {
      if (activeRequest.current === controller) {
        activeRequest.current = null
        setUploading(false)
      }
    }
  }

  const selectBatchRow = (row: number) => {
    const item = batch?.items.find((entry) => entry.row === row)
    if (!item?.result) return
    changeDescription(
      item.name &&
        !item.description
          .toLocaleLowerCase("ru")
          .includes(item.name.toLocaleLowerCase("ru"))
        ? `${item.name}, ${item.description}`
        : item.description,
    )
    setResult(item.result)
    setSelectedBatchRow(row)
    window.requestAnimationFrame(() => {
      resultSection.current?.scrollIntoView({
        behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches
          ? "auto"
          : "smooth",
        block: "start",
      })
    })
  }

  return (
    <div className="app">
      <Header />
      <main className="page">
        <PageHead />
        <SearchForm
          description={description}
          loading={loading}
          onChange={changeDescription}
          onSubmit={startAnalysis}
          onUpload={uploadBatchFile}
          uploading={uploading}
        />
        {batch && (
          <BatchResults
            batch={batch}
            fileName={batchFileName}
            onSelect={selectBatchRow}
            selectedRow={selectedBatchRow}
          />
        )}
        {result && <SearchCorrections corrections={result.corrections} disabled={loading || uploading} onApply={applyCorrection} />}
        {error && (
          <div className="page-notice">
            <NotificationBar title="Не удалось выполнить запрос" tone="danger">
              {error}
            </NotificationBar>
          </div>
        )}
        {(loading || result) && (
          <div className="workspace" ref={resultSection}>
            <div className="workspace__main">
              <ClarifyingQuestion
                loading={loading}
                onAnswer={setSelectedAnswer}
                onSubmitAnswer={submitAnswer}
                result={result}
                selectedAnswer={selectedAnswer}
              />
              <KtruCard loading={loading} result={result} />
            </div>
            <aside className="workspace__side">
              <CharacteristicsCard
                description={description}
                loading={loading && !result}
                onAnalyzeDescription={analyzeEditedDescription}
                onApply={applyCharacteristic}
                result={result}
              />
            </aside>
          </div>
        )}
      </main>
      <Footer />
    </div>
  )
}
