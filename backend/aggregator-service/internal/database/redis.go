// Package database opens connections to the storages used by the aggregator.
package database

import (
	"context"
	"fmt"
	"time"

	"github.com/redis/go-redis/v9"
)

// NewRedis connects to Redis and pings it, so the service fails fast when Redis is unreachable.
func NewRedis(ctx context.Context, url string, pingTimeout time.Duration) (*redis.Client, error) {
	opts, err := redis.ParseURL(url)
	if err != nil {
		return nil, fmt.Errorf("parse REDIS_URL: %w", err)
	}
	// Commands honour context deadlines, so repositories can bound every call.
	opts.ContextTimeoutEnabled = true
	rdb := redis.NewClient(opts)

	ctx, cancel := context.WithTimeout(ctx, pingTimeout)
	defer cancel()
	if err := rdb.Ping(ctx).Err(); err != nil {
		rdb.Close()
		return nil, fmt.Errorf("redis ping: %w", err)
	}
	return rdb, nil
}
