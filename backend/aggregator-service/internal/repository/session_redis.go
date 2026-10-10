package repository

// Sessions in Redis: one JSON value under `session:<id>`. The TTL is renewed on every read and
// write, so inactive sessions expire on their own.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/xqzme-otec/autofilling-key-chatereristics-in-CTE/backend/aggregator-service/internal/domain"
)

const sessionKeyPrefix = "session:"

type RedisSessionRepository struct {
	rdb       *redis.Client
	ttl       time.Duration
	opTimeout time.Duration // bounds every Redis call
}

var _ SessionRepository = (*RedisSessionRepository)(nil)

func NewRedisSessionRepository(rdb *redis.Client, ttl, opTimeout time.Duration) *RedisSessionRepository {
	return &RedisSessionRepository{rdb: rdb, ttl: ttl, opTimeout: opTimeout}
}

func (r *RedisSessionRepository) Get(ctx context.Context, id string) (*domain.Session, error) {
	ctx, cancel := context.WithTimeout(ctx, r.opTimeout)
	defer cancel()
	raw, err := r.rdb.GetEx(ctx, sessionKeyPrefix+id, r.ttl).Bytes()
	if errors.Is(err, redis.Nil) {
		return nil, ErrSessionNotFound
	}
	if err != nil {
		return nil, storageUnavailable(err)
	}
	var session domain.Session
	if err := json.Unmarshal(raw, &session); err != nil {
		return nil, fmt.Errorf("decode session %s: %w", id, err)
	}
	return &session, nil
}

func (r *RedisSessionRepository) Save(ctx context.Context, session *domain.Session) error {
	raw, err := json.Marshal(session)
	if err != nil {
		return fmt.Errorf("encode session %s: %w", session.ID, err)
	}
	ctx, cancel := context.WithTimeout(ctx, r.opTimeout)
	defer cancel()
	if err := r.rdb.Set(ctx, sessionKeyPrefix+session.ID, raw, r.ttl).Err(); err != nil {
		return storageUnavailable(err)
	}
	return nil
}

func storageUnavailable(err error) *domain.Error {
	return domain.Wrap(domain.CodeStorageUnavailable, err, "Хранилище сессий недоступно. Повторите запрос позже")
}
