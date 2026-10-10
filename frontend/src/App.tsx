import { useEffect, useRef, useState } from "react"
import type { ReactNode } from "react"
import { analyze, answerQuestion, ApiError, type AnalysisResult } from "./api"
import { examples } from "./data"
import { Button, Card, Icon, NotificationBar, Spinner, Status } from "./ui"

const PORTAL_URL = "https://zakupki.mos.ru/"

function PortalLink({ children, className, label }: { children: ReactNode; className?: string; label?: string }) {
  return <a aria-label={label} className={className} href={PORTAL_URL} rel="noreferrer" target="_blank">{children}</a>
}

function Header() {
  return <header className="header"><div className="header__inner">
    <PortalLink className="logo" label="Открыть Портал поставщиков">
      <svg aria-hidden="true" height="34" viewBox="0 0 34 34" width="34"><rect fill="#A3A3A3" height="24" width="24" x="0" y="10" /><rect fill="#DB2B21" height="24" width="24" x="8" y="0" /><rect fill="#FFFFFF" height="10" width="10" x="15" y="7" /></svg>
      <span><b>Портал</b><i>Поставщиков</i></span>
    </PortalLink>
    <div className="header__navigation" aria-label="Элементы шапки Портала">
      <PortalLink className="header__item header__menu"><Icon name="menu" size={19} /><span>Меню</span></PortalLink>
      <PortalLink className="header__item header__support"><Icon name="bulb" size={19} /><span>Центр поддержки</span></PortalLink>
      <PortalLink className="header__item header__cart"><Icon name="cart" size={19} /><span>Корзина</span></PortalLink>
      <PortalLink className="header__item header__region"><Icon name="pin" size={19} /><span>Регион</span></PortalLink>
    </div>
    <div className="header__account" aria-label="Элементы учётной записи">
      <PortalLink className="header__search" label="Поиск на Портале"><Icon name="search" size={19} /></PortalLink>
      <span className="header__language" aria-label="Русский язык"><span className="header__flag" /><Icon name="chevronDown" size={12} /></span>
      <PortalLink className="header__login">Войти</PortalLink>
      <PortalLink className="header__register">Зарегистрироваться</PortalLink>
    </div>
  </div></header>
}

function SocialIcon({ name }: { name: "facebook" | "twitter" | "instagram" | "vk" | "youtube" }) {
  const labels = { facebook: "Facebook", twitter: "Twitter", instagram: "Instagram", vk: "ВКонтакте", youtube: "YouTube" }
  return <span aria-label={labels[name]} className="footer__social-icon" role="img">
    <svg aria-hidden="true" viewBox="0 0 64 64">
      {name === "facebook" && <path d="M37 15h9V4h-9C26 4 21 10 21 20v8h-8v11h8v21h12V39h10l2-11H33v-7c0-4 1-6 4-6Z" fill="currentColor" />}
      {name === "twitter" && <path d="M57 16a22 22 0 0 1-7 2 12 12 0 0 0 5-7 24 24 0 0 1-8 3 12 12 0 0 0-21 11C16 24 9 19 4 12a12 12 0 0 0 4 16 12 12 0 0 1-5-2c0 6 4 11 10 12a12 12 0 0 1-5 1c2 5 7 8 12 8A24 24 0 0 1 3 52a34 34 0 0 0 18 5c22 0 34-18 34-34v-2a25 25 0 0 0 6-6l-4 1Z" fill="currentColor" />}
      {name === "instagram" && <><rect fill="currentColor" height="42" rx="7" width="42" x="11" y="11" /><path d="M11 31h42" stroke="var(--red)" strokeWidth="3" /><circle cx="32" cy="32" fill="currentColor" r="12" /><circle cx="32" cy="32" fill="var(--red)" r="8" /><circle cx="46" cy="20" fill="var(--red)" r="3" /></>}
      {name === "vk" && <path d="M7 17h10c1 0 2 1 2 2 2 8 5 14 9 17V19c0-1 1-2 2-2h9c2 0 3 1 3 3v14c4-3 7-9 9-15 0-1 1-2 3-2h9c2 0 3 1 2 3-2 8-7 15-12 20 5 4 9 9 12 16 1 2 0 3-2 3H52c-2 0-3-1-5-3l-7-8v8c0 2-1 3-3 3h-5C18 59 9 45 4 21c0-2 1-4 3-4Z" fill="currentColor" />}
      {name === "youtube" && <><rect fill="currentColor" height="40" rx="10" width="52" x="6" y="12" /><path d="m28 23 16 9-16 9V23Z" fill="var(--red)" /></>}
    </svg>
  </span>
}

function Footer() {
  return <footer className="footer"><div className="footer__inner">
    <div className="footer__main">
      <nav aria-label="Разделы Портала" className="footer__links">
        <PortalLink>О портале</PortalLink><PortalLink>Поставщикам</PortalLink><PortalLink>Покупателям</PortalLink><PortalLink>Новости</PortalLink><PortalLink>Контакты</PortalLink><PortalLink>Карта сайта</PortalLink>
      </nav>
      <p>© 2026 Портал поставщиков<br />Характеристики и подбор КТРУ</p>
    </div>
    <div className="footer__side">
      <div className="footer__help"><PortalLink><Icon name="headset" size={17} /> Служба качества</PortalLink><PortalLink><Icon name="info" size={17} /> Написать в службу поддержки</PortalLink></div>
      <div aria-label="Социальные сети Портала" className="footer__social"><SocialIcon name="facebook" /><SocialIcon name="twitter" /><SocialIcon name="instagram" /><SocialIcon name="vk" /><SocialIcon name="youtube" /></div>
    </div>
  </div></footer>
}

function PageHead() {
  return <div className="page-head">
    <h1>Поиск характеристик и кода КТРУ</h1>
    <p>Введите описание товара или закупочной потребности.</p>
  </div>
}

function SearchForm({ description, onChange, onSubmit, loading }: { description: string; onChange: (value: string) => void; onSubmit: () => void; loading: boolean }) {
  return <form className="search-form" onSubmit={(event) => { event.preventDefault(); onSubmit() }} role="search">
    <label className="sr-only" htmlFor="description">Описание товара или закупочной потребности</label>
    <div className="search-form__row">
      <Icon name="search" size={22} />
      <input autoComplete="off" id="description" maxLength={2000} onChange={(event) => onChange(event.target.value)} placeholder="Например: пальто для девочки, молоток слесарный или офисный стол" type="search" value={description} />
      <Button disabled={!description.trim()} loading={loading} type="submit" variant="primary">{loading ? "Ищем" : "Найти"}</Button>
    </div>
    <div className="search-form__examples"><span>Примеры:</span>{examples.map((item) => <button className="chip" key={item.label} onClick={() => onChange(item.text)} type="button">{item.label}</button>)}</div>
  </form>
}

function CharacteristicsCard({ result, loading }: { result: AnalysisResult | null; loading: boolean }) {
  if (loading) return <Card className="chars"><h2>Распознанные характеристики</h2><div className="api-loading"><Spinner /> Сопоставляем описание со справочником…</div></Card>
  if (!result) return null
  return <Card className="chars" id="characteristics">
    <div className="card__head"><h2>Распознанные характеристики</h2><span className="hint">Найдено: {result.characteristics.length}</span></div>
    {result.corrections.length > 0 && <section className="api-section"><h3>Исправления и нормализация</h3><ul className="api-list">{result.corrections.map((item, index) => <li key={`${item.from}-${index}`}><Icon name="check" size={16} /><span><s>{item.from}</s> → <strong>{item.to}</strong>{item.reason && <small>{item.reason}</small>}</span></li>)}</ul></section>}
    {(result.product_intent || result.subtype_context?.length > 0) && <section className="api-section"><h3>Товар и контекст</h3><p>{result.product_intent && <>Товар: <strong>{result.product_intent}</strong></>}{result.subtype_context?.length > 0 && <> · Контекст: {result.subtype_context.map((signal) => signal.value).join(", ")}</>}</p></section>}
    {result.errors.length > 0 && <section className="api-section" aria-label="Ошибки в описании"><h3>Требует внимания</h3><ul className="api-list api-list--errors">{result.errors.map((item, index) => <li key={`${item.type}-${index}`}><Icon name="alert" size={16} /><span><strong>{item.characteristic || "Описание товара"}</strong>: {item.message}{item.suggestion && <small>Предложение: {item.suggestion}</small>}</span></li>)}</ul></section>}
    {result.characteristics.length > 0 ? <div aria-label="Распознанные характеристики" className="api-table" role="table">
      <div className="api-table__head" role="row"><span role="columnheader">Характеристика</span><span role="columnheader">Значение</span><span role="columnheader">Фрагмент описания</span></div>
      {result.characteristics.map((item, index) => <div className="api-table__row" key={`${item.name}-${index}`} role="row"><strong role="cell">{item.name}</strong><span role="cell">{item.value}{item.unit ? ` ${item.unit}` : ""}</span><span className="api-table__source" role="cell">{({ ktru: "КТРУ", portal: "Портал", user: "Пользователь" } as Record<string, string>)[item.source || ""] || item.source || "—"}{item.user_text && <small>{item.user_text}</small>}</span></div>)}
    </div> : <p className="api-empty">Характеристики пока не найдены. Уточните описание и повторите поиск.</p>}
  </Card>
}

function plural(count: number) {
  const mod10 = count % 10, mod100 = count % 100
  if (mod10 === 1 && mod100 !== 11) return "позиция"
  if (mod10 >= 2 && mod10 <= 4 && (mod100 < 12 || mod100 > 14)) return "позиции"
  return "позиций"
}

function KtruCard({ result, loading, selectedAnswer, onAnswer, onSubmitAnswer }: { result: AnalysisResult | null; loading: boolean; selectedAnswer: string; onAnswer: (value: string) => void; onSubmitAnswer: () => void }) {
  if (loading && !result) return <Card className="ktru"><div className="ktru-head"><h2>Подбор кода КТРУ</h2></div><div className="api-loading"><Spinner /> Ищем позиции…</div></Card>
  if (!result) return <Card className="ktru"><div className="ktru-head"><h2>Подбор кода КТРУ</h2><Status tone="neutral">Ожидает данных</Status></div><div className="ktru-empty"><p>Подбор начнётся после анализа описания.</p></div></Card>
  const complete = result.ktru !== null
  const lastCount = result.remaining_count ?? result.steps.at(-1)?.count ?? result.candidates.length
  return <div className="api-side-stack"><Card className="ktru">
    <div className="ktru-head"><div className="ktru-head__row"><h2>Подбор кода КТРУ</h2><Status tone={complete ? "success" : result.next_step ? "info" : "warning"}>{complete ? "Код определён" : result.next_step ? "Нужно уточнение" : "Нет результата"}</Status></div>{!complete && <p className="ktru-head__count">Поиск: {result.retrieval_count ?? result.candidates.length} · По товару: {result.compatible_count ?? lastCount} · Осталось: <b>{lastCount}</b> {plural(lastCount)}</p>}</div>
    {result.steps.length > 0 && <ol aria-label="Сужение списка КТРУ" className="history">{result.steps.map((step, index) => <li className={index === result.steps.length - 1 ? "is-last" : ""} key={`${step.stage}-${index}`}><span className="history__dot" /><span className="history__label">{step.stage}</span><span className="history__count">{step.count}</span></li>)}</ol>}
    {complete && <div className="api-final"><span className="label-caps">Позиция КТРУ</span><h3>{result.ktru_name || "Товар"}</h3><strong className="api-code">{result.ktru}</strong></div>}
    {!complete && !result.next_step && <p className="api-empty">Однозначный код не найден. Уточните описание и запустите подбор заново.</p>}
    {!complete && result.candidates.length > 0 && <div className="api-candidates"><h3>Подходящие позиции</h3><ul>{result.candidates.map((candidate) => <li key={candidate.code}><strong>{candidate.name}</strong><span>{candidate.code}</span>{candidate.score != null && <small>Совпадение: {Math.round(candidate.score * 100)}%</small>}</li>)}</ul></div>}

  </Card>
    {result.next_step && <Card className="question" id="question"><span className="label-caps label-caps--blue">Уточняющий вопрос</span><h3>{result.next_step.message}</h3>{result.next_step.characteristic && <p>Характеристика: {result.next_step.characteristic}</p>}
      <fieldset className="radios" disabled={loading}><legend className="sr-only">{result.next_step.message}</legend>{result.next_step.options.map((option) => <label className={`radio ${selectedAnswer === option ? "is-checked" : ""}`} key={option}><input checked={selectedAnswer === option} name="clarify" onChange={() => onAnswer(option)} type="radio" value={option} /><span className="radio__mark" />{option}</label>)}</fieldset>
      <Button className="btn--block" disabled={!selectedAnswer} loading={loading} onClick={onSubmitAnswer} variant="primary">{loading ? "Уточняем подбор" : "Уточнить подбор"}</Button>
    </Card>}
  </div>
}

export default function App() {
  const [description, setDescription] = useState(() => { try { return localStorage.getItem("ste-description") || "" } catch { return "" } })
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [selectedAnswer, setSelectedAnswer] = useState("")
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")
  const activeRequest = useRef<AbortController | null>(null)

  useEffect(() => { try { localStorage.setItem("ste-description", description) } catch { /* Storage may be unavailable. */ } }, [description])
  useEffect(() => () => activeRequest.current?.abort(), [])

  const changeDescription = (value: string) => {
    activeRequest.current?.abort()
    activeRequest.current = null
    setLoading(false)
    setDescription(value)
    setResult(null)
    setSelectedAnswer("")
    setError("")
  }

  const perform = async (request: (signal: AbortSignal) => Promise<AnalysisResult>) => {
    activeRequest.current?.abort()
    const controller = new AbortController()
    activeRequest.current = controller
    setLoading(true)
    setError("")
    try {
      const next = await request(controller.signal)
      if (activeRequest.current !== controller) return
      setResult(next)
      setSelectedAnswer("")
    } catch (cause) {
      if (controller.signal.aborted || activeRequest.current !== controller) return
      if (cause instanceof ApiError) {
        setError(cause.code === "session_not_found" ? "Сессия больше не доступна. Запустите подбор заново по сохранённому описанию." : cause.code === "model_unavailable" || cause.code === "model_timeout" || cause.code === "model_bad_response" ? "Модельный сервис пока не ответил. Повторите запрос позже." : cause.message)
      } else setError("Не удалось обработать запрос. Попробуйте ещё раз.")
    } finally {
      if (activeRequest.current === controller) { activeRequest.current = null; setLoading(false) }
    }
  }

  const startAnalysis = () => { if (!description.trim()) return; setResult(null); setSelectedAnswer(""); void perform((signal) => analyze(description, signal)) }
  const submitAnswer = () => { if (!result?.next_step || !result.next_step.options.includes(selectedAnswer)) return; void perform((signal) => answerQuestion(result.session_id, selectedAnswer, signal)) }

  return <div className="app"><Header /><main className="page"><PageHead />
    <SearchForm description={description} loading={loading} onChange={changeDescription} onSubmit={startAnalysis} />
    {error && <div className="page-notice"><NotificationBar title="Не удалось выполнить запрос" tone="danger">{error}</NotificationBar></div>}
    {(loading || result) && <div className="workspace"><div className="workspace__main"><CharacteristicsCard loading={loading && !result} result={result} /></div><aside className="workspace__side"><KtruCard loading={loading} onAnswer={setSelectedAnswer} onSubmitAnswer={submitAnswer} result={result} selectedAnswer={selectedAnswer} /></aside></div>}
  </main><Footer /></div>
}
