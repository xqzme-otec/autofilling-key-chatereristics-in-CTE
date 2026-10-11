export interface Correction {
  from: string
  to: string
  reason?: string | null
}

export interface Characteristic {
  name: string
  value: string
  unit?: string | null
  source?: string | null
  origin?: "ktru" | "portal" | "new" | "answer" | null
  known?: boolean
  user_text?: string | null
  used_for_narrowing?: boolean
  used_for_reranking?: boolean
}

export interface ContextSignal {
  value: string
  source: string
  user_text: string
  used_for_narrowing: boolean
  used_for_reranking: boolean
}

export interface Candidate {
  code: string
  name: string
  score?: number | null
}

export interface AnalysisError {
  type: string
  severity?: "error" | "warning" | "info"
  message: string
  characteristic?: string | null
  suggestion?: string | null
}

export interface NextStep {
  message: string
  characteristic: string | null
  options: string[]
}

export interface AnalysisResult {
  session_id: string
  product_intent: string | null
  subtype_context: ContextSignal[]
  retrieval_count: number
  compatible_count: number
  remaining_count: number
  corrections: Correction[]
  characteristics: Characteristic[]
  candidates: Candidate[]
  errors: AnalysisError[]
  missing_required?: string[]
  next_step: NextStep | null
  next_question?: NextStep | null
  ktru: string | null
  ktru_name: string | null
  steps: { stage: string; count: number }[]
}

export interface BatchItem {
  row: number
  name?: string
  description: string
  result?: AnalysisResult
  error?: { code: string; message: string }
}

export interface BatchResult {
  items: BatchItem[]
}

interface ErrorResponse {
  error?: { code?: string; message?: string; request_id?: string }
}

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly code: string,
    public readonly status: number,
    public readonly requestId?: string,
  ) {
    super(message)
    this.name = "ApiError"
  }
}

const localHosts = new Set(["localhost", "127.0.0.1", "[::1]", "::1"])
const baseUrl = localHosts.has(window.location.hostname)
  ? ""
  : (import.meta.env.VITE_API_BASE_URL || "").trim().replace(/\/+$/, "")

async function requestJson(
  path: string,
  options: RequestInit,
): Promise<{ body: unknown; status: number; requestId?: string }> {
  let response: Response
  try {
    response = await fetch(`${baseUrl}${path}`, options)
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError")
      throw error
    throw new ApiError(
      "Не удалось подключиться к агрегатору. Проверьте, что сервис запущен.",
      "network_error",
      0,
    )
  }

  const body: unknown = await response.json().catch(() => null)
  const headerRequestId = response.headers.get("X-Request-ID") || undefined
  if (!response.ok) {
    const details = body as ErrorResponse | null
    const code = details?.error?.code || "http_error"
    const message =
      details?.error?.message || `Сервис вернул ошибку ${response.status}`
    const requestId = details?.error?.request_id || headerRequestId
    throw new ApiError(message, code, response.status, requestId)
  }

  return { body, status: response.status, requestId: headerRequestId }
}

async function post(
  path: string,
  payload: object,
  signal?: AbortSignal,
): Promise<AnalysisResult> {
  const { body, status, requestId } = await requestJson(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  })
  if (!isAnalysisResult(body)) {
    throw new ApiError(
      "Агрегатор вернул неожиданный ответ.",
      "bad_response",
      status,
      requestId,
    )
  }
  return body
}

function isAnalysisResult(value: unknown): value is AnalysisResult {
  if (!value || typeof value !== "object") return false
  const result = value as Partial<AnalysisResult>
  return (
    typeof result.session_id === "string" &&
    Array.isArray(result.corrections) &&
    Array.isArray(result.characteristics) &&
    Array.isArray(result.candidates) &&
    Array.isArray(result.errors) &&
    Array.isArray(result.steps) &&
    (result.next_step === null ||
      (typeof result.next_step === "object" &&
        Array.isArray(result.next_step?.options))) &&
    (result.ktru === null || typeof result.ktru === "string")
  )
}

export function analyze(text: string, signal?: AbortSignal) {
  return post("/api/v1/analyze", { text }, signal)
}

export function answerQuestion(
  sessionId: string,
  value: string,
  signal?: AbortSignal,
) {
  return post(
    `/api/v1/sessions/${encodeURIComponent(sessionId)}/answers`,
    { value },
    signal,
  )
}

export async function analyzeBatch(
  file: File,
  signal?: AbortSignal,
): Promise<BatchResult> {
  const form = new FormData()
  form.append("file", file)
  const { body, status, requestId } = await requestJson("/api/v1/analyze/batch", {
    method: "POST",
    body: form,
    signal,
  })
  if (
    !body ||
    typeof body !== "object" ||
    !Array.isArray((body as Partial<BatchResult>).items) ||
    !(body as BatchResult).items.every(
      (item) =>
        Number.isInteger(item.row) &&
        typeof item.description === "string" &&
        ((item.result && isAnalysisResult(item.result)) ||
          (item.error && typeof item.error.message === "string")),
    )
  ) {
    throw new ApiError(
      "Агрегатор вернул неожиданный ответ для таблицы.",
      "bad_response",
      status,
      requestId,
    )
  }
  return body as BatchResult
}
