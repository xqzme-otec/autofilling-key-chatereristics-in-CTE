"""New characteristics in Postgres: the level of the source hierarchy between the supplier portal and
a brand-new characteristic (KTRU → portal → seen earlier → new, docs/DOC.md, section 5).

One row per (category, product, characteristic key, description). The next round of a dialog sends
the same description again, so it is not a new sighting: a characteristic counts as seen earlier
only when a different description of the same product had it.

The table is created by the service itself (CREATE TABLE IF NOT EXISTS on every new connection), so
an empty or recreated database needs no migration step.
"""
from __future__ import annotations

import logging

import psycopg
from psycopg_pool import ConnectionPool

logger = logging.getLogger(__name__)

# waiting for a pooled connection; with Postgres down a request pays it once, then the step is skipped
CONNECTION_TIMEOUT_S = 1.0

SCHEMA = """
CREATE TABLE IF NOT EXISTS new_characteristics (
    category   text        NOT NULL,
    product    text        NOT NULL,
    key        text        NOT NULL,
    text_hash  text        NOT NULL,
    name       text        NOT NULL,
    value      text        NOT NULL,
    unit       text,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (category, product, key, text_hash)
)
"""


def _create_schema(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA)
    conn.commit()  # the pool takes the connection back only when it is idle


class NewCharacteristicsRepository:
    def __init__(self, dsn: str):
        # open=False + open(wait=False): connecting happens in the background and is retried, so the
        # service starts and works (degraded) while Postgres is down. max_size=4 is plenty: the pipeline
        # lock lets requests reach this step one at a time, and the step takes a few ms
        self.pool = ConnectionPool(dsn, min_size=1, max_size=4, open=False, timeout=CONNECTION_TIMEOUT_S,
                                   kwargs={"connect_timeout": 3}, configure=_create_schema,
                                   check=ConnectionPool.check_connection, name="new-characteristics")

    def open(self) -> None:
        self.pool.open(wait=False)

    def close(self) -> None:
        self.pool.close()

    def ping(self) -> bool:
        try:
            with self.pool.connection() as conn:
                conn.execute("SELECT 1")
        except psycopg.Error as e:  # PoolTimeout is a psycopg.OperationalError too
            logger.debug("postgres ping failed: %s", e)
            return False
        return True

    def seen_before(self, category: str, product: str, keys: list[str], text_hash: str) -> dict[str, int]:
        """key -> number of other descriptions of this product that had it. Raises psycopg.Error."""
        with self.pool.connection() as conn:
            rows = conn.execute(
                "SELECT key, count(*) FROM new_characteristics"
                " WHERE category = %s AND product = %s AND key = ANY(%s) AND text_hash <> %s GROUP BY key",
                (category, product, keys, text_hash)).fetchall()
        return dict(rows)

    def save(self, category: str, product: str, text_hash: str,
             items: list[tuple[str, str, str, str | None]]) -> None:
        """Saves (key, name, value, unit) items of one description; the same description again adds
        nothing. Raises psycopg.Error."""
        if not items:
            return
        with self.pool.connection() as conn, conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO new_characteristics (category, product, key, text_hash, name, value, unit)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                [(category, product, key, text_hash, name, value, unit) for key, name, value, unit in items])
