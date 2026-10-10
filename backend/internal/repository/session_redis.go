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

	"github.com/xqzme-otec/autofilling-key-chatereristics-in-CTE/backend/internal/domain"
)

const (
	sessionKeyPrefix = "session:"
	// redisOpTimeout bounds every Redis call: when Redis is down, go-redis keeps retrying dials
	// for over a minute and the request would hang.
	redisOpTimeout = 3 * time.Second
)

type RedisSessionRepository struct {
	rdb *redis.Client
	ttl time.Duration
}

var _ SessionRepository = (*RedisSessionRepository)(nil)

func NewRedisSessionRepository(rdb *redis.Client, ttl time.Duration) *RedisSessionRepository {
	return &RedisSessionRepository{rdb: rdb, ttl: ttl}
}

func (r *RedisSessionRepository) Get(ctx context.Context, id string) (*domain.Session, error) {
	ctx, cancel := context.WithTimeout(ctx, redisOpTimeout)
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
	ctx, cancel := context.WithTimeout(ctx, redisOpTimeout)
	defer cancel()
	if err := r.rdb.Set(ctx, sessionKeyPrefix+session.ID, raw, r.ttl).Err(); err != nil {
		return storageUnavailable(err)
	}
	return nil
}

func storageUnavailable(err error) *domain.Error {
	return domain.Errorf(domain.CodeStorageUnavailable, "Session storage is unavailable: %v", err)
}
