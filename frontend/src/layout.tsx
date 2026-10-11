import type { ReactNode } from "react"
import { Icon } from "./ui"

const PORTAL_URL = "https://zakupki.mos.ru/"

function PortalLink({
  children,
  className,
  label,
}: {
  children: ReactNode
  className?: string
  label?: string
}) {
  return (
    <a
      aria-label={label}
      className={className}
      href={PORTAL_URL}
      rel="noreferrer"
      target="_blank"
    >
      {children}
    </a>
  )
}

export function Header() {
  return (
    <header className="header">
      <div className="header__inner">
        <PortalLink className="logo" label="Открыть Портал поставщиков">
          <svg aria-hidden="true" height="34" viewBox="0 0 34 34" width="34">
            <rect fill="#A3A3A3" height="24" width="24" x="0" y="10" />
            <rect fill="#DB2B21" height="24" width="24" x="8" y="0" />
            <rect fill="#FFFFFF" height="10" width="10" x="15" y="7" />
          </svg>
          <span>
            <b>Портал</b>
            <i>Поставщиков</i>
          </span>
        </PortalLink>
        <div className="header__navigation" aria-label="Элементы шапки Портала">
          <PortalLink className="header__item header__menu">
            <Icon name="menu" size={19} />
            <span>Меню</span>
          </PortalLink>
          <PortalLink className="header__item header__support">
            <Icon name="bulb" size={19} />
            <span>Центр поддержки</span>
          </PortalLink>
          <PortalLink className="header__item header__cart">
            <Icon name="cart" size={19} />
            <span>Корзина</span>
          </PortalLink>
          <PortalLink className="header__item header__region">
            <Icon name="pin" size={19} />
            <span>Регион</span>
          </PortalLink>
        </div>
        <div className="header__account" aria-label="Элементы учётной записи">
          <PortalLink className="header__search" label="Поиск на Портале">
            <Icon name="search" size={19} />
          </PortalLink>
          <span className="header__language" aria-label="Русский язык">
            <span className="header__flag" />
            <Icon name="chevronDown" size={12} />
          </span>
          <PortalLink className="header__login">Войти</PortalLink>
          <PortalLink className="header__register">
            Зарегистрироваться
          </PortalLink>
        </div>
      </div>
    </header>
  )
}

function SocialIcon({
  name,
}: {
  name: "facebook" | "twitter" | "instagram" | "vk" | "youtube"
}) {
  const labels = {
    facebook: "Facebook",
    twitter: "Twitter",
    instagram: "Instagram",
    vk: "ВКонтакте",
    youtube: "YouTube",
  }
  return (
    <span aria-label={labels[name]} className="footer__social-icon" role="img">
      <svg aria-hidden="true" viewBox="0 0 64 64">
        {name === "facebook" && (
          <path
            d="M37 15h9V4h-9C26 4 21 10 21 20v8h-8v11h8v21h12V39h10l2-11H33v-7c0-4 1-6 4-6Z"
            fill="currentColor"
          />
        )}
        {name === "twitter" && (
          <path
            d="M57 16a22 22 0 0 1-7 2 12 12 0 0 0 5-7 24 24 0 0 1-8 3 12 12 0 0 0-21 11C16 24 9 19 4 12a12 12 0 0 0 4 16 12 12 0 0 1-5-2c0 6 4 11 10 12a12 12 0 0 1-5 1c2 5 7 8 12 8A24 24 0 0 1 3 52a34 34 0 0 0 18 5c22 0 34-18 34-34v-2a25 25 0 0 0 6-6l-4 1Z"
            fill="currentColor"
          />
        )}
        {name === "instagram" && (
          <>
            <rect
              fill="currentColor"
              height="42"
              rx="7"
              width="42"
              x="11"
              y="11"
            />
            <path d="M11 31h42" stroke="var(--red)" strokeWidth="3" />
            <circle cx="32" cy="32" fill="currentColor" r="12" />
            <circle cx="32" cy="32" fill="var(--red)" r="8" />
            <circle cx="46" cy="20" fill="var(--red)" r="3" />
          </>
        )}
        {name === "vk" && (
          <path
            d="M7 17h10c1 0 2 1 2 2 2 8 5 14 9 17V19c0-1 1-2 2-2h9c2 0 3 1 3 3v14c4-3 7-9 9-15 0-1 1-2 3-2h9c2 0 3 1 2 3-2 8-7 15-12 20 5 4 9 9 12 16 1 2 0 3-2 3H52c-2 0-3-1-5-3l-7-8v8c0 2-1 3-3 3h-5C18 59 9 45 4 21c0-2 1-4 3-4Z"
            fill="currentColor"
          />
        )}
        {name === "youtube" && (
          <>
            <rect
              fill="currentColor"
              height="40"
              rx="10"
              width="52"
              x="6"
              y="12"
            />
            <path d="m28 23 16 9-16 9V23Z" fill="var(--red)" />
          </>
        )}
      </svg>
    </span>
  )
}

export function Footer() {
  return (
    <footer className="footer">
      <div className="footer__inner">
        <div className="footer__main">
          <nav aria-label="Разделы Портала" className="footer__links">
            <PortalLink>О портале</PortalLink>
            <PortalLink>Поставщикам</PortalLink>
            <PortalLink>Покупателям</PortalLink>
            <PortalLink>Новости</PortalLink>
            <PortalLink>Контакты</PortalLink>
            <PortalLink>Карта сайта</PortalLink>
          </nav>
          <p>
            © 2026 Портал поставщиков
            <br />
            Характеристики и подбор КТРУ
          </p>
        </div>
        <div className="footer__side">
          <div className="footer__help">
            <PortalLink>
              <Icon name="headset" size={17} /> Служба качества
            </PortalLink>
            <PortalLink>
              <Icon name="info" size={17} /> Написать в службу поддержки
            </PortalLink>
          </div>
          <div aria-label="Социальные сети Портала" className="footer__social">
            <SocialIcon name="facebook" />
            <SocialIcon name="twitter" />
            <SocialIcon name="instagram" />
            <SocialIcon name="vk" />
            <SocialIcon name="youtube" />
          </div>
        </div>
      </div>
    </footer>
  )
}

export function PageHead() {
  return (
    <div className="page-head">
      <h1>Поиск характеристик и кода КТРУ</h1>
      <p>Введите описание товара или закупочной потребности.</p>
    </div>
  )
}

