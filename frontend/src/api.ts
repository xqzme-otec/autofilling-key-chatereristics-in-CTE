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
  next_step: NextStep | null
  next_question?: NextStep | null
  ktru: string | null
  ktru_name: string | null
  steps: { stage: string; count: number }[]
}

interface ErrorResponse {
  error?: { code?: string; message?: string }
}

export class ApiError extends Error {
  constructor(message: string, public readonly code: string, public readonly status: number) {
    super(message)
    this.name = "ApiError"
  }
}

const baseUrl = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "")

async function post(path: string, payload: object, signal?: AbortSignal): Promise<AnalysisResult> {
  let response: Response
  try {
    response = await fetch(`${baseUrl}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal,
    })
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw error
    throw new ApiError("Не удалось подключиться к агрегатору. Проверьте, что сервис запущен.", "network_error", 0)
  }

  const body: unknown = await response.json().catch(() => null)
  if (!response.ok) {
    const details = body as ErrorResponse | null
    const code = details?.error?.code || "http_error"
    const message = details?.error?.message || `Сервис вернул ошибку ${response.status}`
    throw new ApiError(message, code, response.status)
  }

  if (!isAnalysisResult(body)) {
    throw new ApiError("Агрегатор вернул неожиданный ответ.", "bad_response", response.status)
  }
  return body
}

function isAnalysisResult(value: unknown): value is AnalysisResult {
  if (!value || typeof value !== "object") return false
  const result = value as Partial<AnalysisResult>
  return typeof result.session_id === "string"
    && Array.isArray(result.corrections)
    && Array.isArray(result.characteristics)
    && Array.isArray(result.candidates)
    && Array.isArray(result.errors)
    && Array.isArray(result.steps)
    && (result.next_step === null || (typeof result.next_step === "object" && Array.isArray(result.next_step?.options)))
    && (result.ktru === null || typeof result.ktru === "string")
}

export function analyze(text: string, signal?: AbortSignal) {
  return post("/api/v1/analyze", { text }, signal)
}

export function answerQuestion(sessionId: string, value: string, signal?: AbortSignal) {
  return post(`/api/v1/sessions/${encodeURIComponent(sessionId)}/answers`, { value }, signal)
}
