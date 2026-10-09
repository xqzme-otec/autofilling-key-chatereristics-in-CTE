import type { ReactNode } from "react"
import type { Check, Source } from "./data"

export type IconName =
  | "alert"
  | "bell"
  | "bulb"
  | "catalog"
  | "cart"
  | "check"
  | "chevronDown"
  | "chevronRight"
  | "close"
  | "doc"
  | "edit"
  | "info"
  | "headset"
  | "menu"
  | "pin"
  | "plus"
  | "refresh"
  | "search"
  | "server"
  | "trash"

const paths: Record<IconName, ReactNode> = {
  alert: (
    <>
      <path d="M10.3 4.2 2.6 18a2 2 0 0 0 1.7 3h15.4a2 2 0 0 0 1.7-3L13.7 4.2a2 2 0 0 0-3.4 0Z" />
      <path d="M12 9.5v4M12 17h.01" />
    </>
  ),
  bell: (
    <>
      <path d="M6 16V11a6 6 0 1 1 12 0v5l1.5 2h-15Z" />
      <path d="M10 20.5a2 2 0 0 0 4 0" />
    </>
  ),
  bulb: (
    <>
      <path d="M9 17h6M10 20.5h4" />
      <path d="M8.2 13.8A5.5 5.5 0 1 1 15.8 13.8c-.8.8-1.3 1.7-1.3 2.7h-5c0-1-.5-1.9-1.3-2.7Z" />
      <path d="M12 1.5v1.5M3.5 9H2M22 9h-1.5M5.5 3.5l1 1M18.5 3.5l-1 1" />
    </>
  ),
  catalog: (
    <>
      <rect height="7" width="7" x="3.5" y="3.5" />
      <rect height="7" width="7" x="13.5" y="3.5" />
      <rect height="7" width="7" x="3.5" y="13.5" />
      <rect height="7" width="7" x="13.5" y="13.5" />
    </>
  ),
  cart: (
    <>
      <circle cx="9" cy="20" r="1" />
      <circle cx="18" cy="20" r="1" />
      <path d="M2 4h2l2.3 11.5h12.8l2-8.5H5" />
    </>
  ),
  check: <path d="m5 12.5 4.2 4.2L19 7" />,
  chevronDown: <path d="m7 10 5 5 5-5" />,
  chevronRight: <path d="m10 7 5 5-5 5" />,
  close: <path d="m6 6 12 12M18 6 6 18" />,
  doc: (
    <>
      <path d="M6 2.5h8.5L19 7v14.5H6Z" />
      <path d="M14 2.5V7h5M9 12h7M9 15.5h7M9 19h4" />
    </>
  ),
  edit: (
    <>
      <path d="M4 20h4L19 9l-4-4L4 16Z" />
      <path d="m13.5 6.5 4 4" />
    </>
  ),
  info: (
    <>
      <circle cx="12" cy="12" r="9" />
      <path d="M12 11v5.5M12 7.6h.01" />
    </>
  ),
  headset: (
    <>
      <path d="M4 13v-2a8 8 0 0 1 16 0v2M4 13h3v6H5a2 2 0 0 1-2-2v-2a2 2 0 0 1 1-2ZM20 13h-3v6h2a2 2 0 0 0 2-2v-2a2 2 0 0 0-1-2ZM17 19a5 5 0 0 1-5 3h-2" />
    </>
  ),
  menu: <path d="M3 6h18M3 12h18M3 18h18" />,
  pin: <path d="M21 3 3 10.5l7.5 3 3 7.5Z" />,
  plus: <path d="M12 5v14M5 12h14" />,
  refresh: (
    <>
      <path d="M20 12a8 8 0 1 1-2.4-5.7" />
      <path d="M20 4v5h-5" />
    </>
  ),
  search: (
    <>
      <circle cx="10.5" cy="10.5" r="7" />
      <path d="m20.5 20.5-5-5" />
    </>
  ),
  server: (
    <>
      <rect height="5" width="16" x="4" y="3.5" />
      <rect height="5" width="16" x="4" y="10.5" />
      <path d="M4 17.5h8M7 6h.01M7 13h.01" />
      <circle cx="17.5" cy="19" r="3" />
      <path d="m15.4 21.1 4.2-4.2" />
    </>
  ),
  trash: (
    <>
      <path d="M4 6.5h16M9.5 6.5V4h5v2.5M6.5 6.5l1 14h9l1-14" />
      <path d="M10 10.5v6M14 10.5v6" />
    </>
  ),
}

export function Icon({ name, size = 16, className = "" }: { name: IconName; size?: number; className?: string }) {
  return (
    <svg aria-hidden="true" className={`icon ${className}`} fill="none" height={size} viewBox="0 0 24 24" width={size}>
      {paths[name]}
    </svg>
  )
}

export function Spinner({ light = false }: { light?: boolean }) {
  return <span aria-hidden="true" className={light ? "spinner spinner--light" : "spinner"} />
}

export function Button({
  children,
  icon,
  onClick,
  variant = "secondary",
  size = "md",
  disabled = false,
  loading = false,
  className = "",
  type = "button",
}: {
  children: ReactNode
  icon?: IconName
  onClick?: () => void
  variant?: "primary" | "secondary" | "outline" | "link" | "inverse"
  size?: "md" | "sm"
  disabled?: boolean
  loading?: boolean
  className?: string
  type?: "button" | "submit"
}) {
  return (
    <button
      aria-busy={loading || undefined}
      className={`btn btn--${variant} btn--${size} ${loading ? "is-loading" : ""} ${className}`}
      disabled={disabled || loading}
      onClick={onClick}
      type={type}
    >
      {loading ? <Spinner light={variant === "primary"} /> : icon && <Icon name={icon} />}
      <span>{children}</span>
    </button>
  )
}

export function IconButton({
  icon,
  label,
  onClick,
  active = false,
}: {
  icon: IconName
  label: string
  onClick?: () => void
  active?: boolean
}) {
  return (
    <button
      aria-expanded={icon === "chevronDown" ? active : undefined}
      aria-label={label}
      className={`icon-btn ${active ? "is-active" : ""}`}
      onClick={onClick}
      title={label}
      type="button"
    >
      <Icon name={icon} />
    </button>
  )
}

export function SourceBadge({ source }: { source: Source }) {
  const tone = source === "КТРУ" ? "ktru" : source === "Портал" ? "portal" : "user"
  return <span className={`source source--${tone}`}>{source}</span>
}

export type StatusTone = "success" | "danger" | "warning" | "info" | "neutral"

export function Status({ tone, children }: { tone: StatusTone; children: ReactNode }) {
  const icon: IconName = tone === "success" ? "check" : tone === "info" ? "info" : tone === "neutral" ? "info" : "alert"
  return (
    <span className={`status status--${tone}`}>
      <span className="status__icon">
        <Icon name={icon} size={10} />
      </span>
      {children}
    </span>
  )
}

export function CheckStatus({ check }: { check: Check }) {
  switch (check) {
    case "matched":
      return <Status tone="success">Сопоставлено</Status>
    case "synonym":
      return <Status tone="success">Синоним сопоставлен</Status>
    case "edited":
      return <Status tone="info">Изменено вручную</Status>
    case "error":
      return <Status tone="danger">Ошибка</Status>
    case "typo":
      return <Status tone="warning">Возможна опечатка</Status>
  }
}

/** Плашка уведомления в стиле компонента Notification из UI-кита */
export function NotificationBar({
  tone,
  title,
  children,
  actions,
}: {
  tone: "success" | "danger" | "warning" | "info"
  title: ReactNode
  children?: ReactNode
  actions?: ReactNode
}) {
  return (
    <div className={`notification notification--${tone}`} role={tone === "danger" ? "alert" : "status"}>
      <span className="notification__icon">
        <Icon name={tone === "success" ? "check" : tone === "info" ? "info" : "alert"} size={14} />
      </span>
      <div className="notification__text">
        <strong>{title}</strong>
        {children && <p>{children}</p>}
      </div>
      {actions && <div className="notification__actions">{actions}</div>}
    </div>
  )
}

export function Card({ children, className = "", id }: { children: ReactNode; className?: string; id?: string }) {
  return (
    <section className={`card ${className}`} id={id}>
      {children}
    </section>
  )
}

export function Placeholder({ small = false }: { small?: boolean }) {
  return (
    <div aria-hidden="true" className={small ? "placeholder placeholder--sm" : "placeholder"}>
      <span />
      <span />
    </div>
  )
}
