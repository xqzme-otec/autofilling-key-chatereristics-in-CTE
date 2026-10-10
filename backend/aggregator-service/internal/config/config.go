// Package config reads the service settings from environment variables.
package config

import (
	"errors"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
	"time"

	"go.uber.org/zap/zapcore"
)

type Config struct {
	LogLevel          zapcore.Level
	ListenAddr        string
	ShutdownTimeout   time.Duration
	ReadHeaderTimeout time.Duration
	MaxBodyBytes      int64
	ModelServiceURL   string
	ModelTimeout      time.Duration
	MaxTextLength     int
	MaxBatchItems     int
	BatchConcurrency  int
	RedisURL          string
	RedisPingTimeout  time.Duration
	RedisOpTimeout    time.Duration
	SessionTTL        time.Duration
	CORSOrigins       []string
}

func Load() (Config, error) {
	var errs []error
	sec := func(key string, def float64) time.Duration {
		d, err := seconds(key, def)
		errs = append(errs, err)
		return d
	}
	num := func(key string, def int) int {
		n, err := positiveInt(key, def)
		errs = append(errs, err)
		return n
	}
	logLevel, err := zapcore.ParseLevel(env("LOG_LEVEL", "info"))
	if err != nil {
		errs = append(errs, fmt.Errorf("LOG_LEVEL must be debug, info, warn or error, got %q", os.Getenv("LOG_LEVEL")))
	}
	var origins []string
	for o := range strings.SplitSeq(env("CORS_ORIGINS", "*"), ",") {
		if o = strings.TrimSpace(o); o != "" {
			origins = append(origins, o)
		}
	}
	cfg := Config{
		LogLevel:   logLevel,
		ListenAddr: env("LISTEN_ADDR", ":8000"),
		// Keep under the container's stop grace period (stop_grace_period in docker-compose.yaml).
		ShutdownTimeout:   sec("SHUTDOWN_TIMEOUT_S", 8),
		ReadHeaderTimeout: sec("READ_HEADER_TIMEOUT_S", 10),
		MaxBodyBytes:      int64(num("MAX_BODY_BYTES", 1<<20)),
		ModelServiceURL:   strings.TrimRight(env("MODEL_SERVICE_URL", "http://localhost:8001"), "/"),
		ModelTimeout:      sec("MODEL_TIMEOUT_S", 30),
		MaxTextLength:     num("MAX_TEXT_LENGTH", 2000),
		// A table is analyzed within one request: keep the batch small enough for the proxy
		// in front of the API to wait for (proxy_read_timeout in frontend/nginx.conf).
		MaxBatchItems:    num("MAX_BATCH_ITEMS", 50),
		BatchConcurrency: num("BATCH_CONCURRENCY", 4),
		RedisURL:         env("REDIS_URL", "redis://localhost:6379/0"),
		RedisPingTimeout: sec("REDIS_PING_TIMEOUT_S", 5),
		// Bounds every Redis call: when Redis is down, go-redis keeps retrying dials for over a minute.
		RedisOpTimeout: sec("REDIS_OP_TIMEOUT_S", 3),
		SessionTTL:     time.Duration(num("SESSION_TTL_S", 3600)) * time.Second,
		CORSOrigins:    origins,
	}
	if err := errors.Join(errs...); err != nil {
		return Config{}, err
	}
	return cfg, nil
}

func env(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

// seconds parses a positive, possibly fractional number of seconds.
func seconds(key string, def float64) (time.Duration, error) {
	n := def
	if v := os.Getenv(key); v != "" {
		var err error
		// !(n > 0) also rejects NaN.
		if n, err = strconv.ParseFloat(v, 64); err != nil || !(n > 0) || math.IsInf(n, 0) {
			return 0, fmt.Errorf("%s must be a positive number of seconds, got %q", key, v)
		}
	}
	return time.Duration(n * float64(time.Second)), nil
}

func positiveInt(key string, def int) (int, error) {
	v := os.Getenv(key)
	if v == "" {
		return def, nil
	}
	n, err := strconv.Atoi(v)
	if err != nil || n <= 0 {
		return 0, fmt.Errorf("%s must be a positive integer, got %q", key, v)
	}
	return n, nil
}
