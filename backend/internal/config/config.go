// Package config reads the service settings from environment variables.
package config

import (
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
	"time"
)

type Config struct {
	ModelServiceURL string
	ModelTimeout    time.Duration
	MaxTextLength   int
	RedisURL        string
	SessionTTL      time.Duration
	CORSOrigins     []string
}

func Load() (Config, error) {
	modelTimeout, err := seconds("MODEL_TIMEOUT_S", 30)
	if err != nil {
		return Config{}, err
	}
	maxTextLength, err := positiveInt("MAX_TEXT_LENGTH", 2000)
	if err != nil {
		return Config{}, err
	}
	sessionTTL, err := positiveInt("SESSION_TTL_S", 3600)
	if err != nil {
		return Config{}, err
	}
	var origins []string
	for o := range strings.SplitSeq(env("CORS_ORIGINS", "*"), ",") {
		if o = strings.TrimSpace(o); o != "" {
			origins = append(origins, o)
		}
	}
	return Config{
		ModelServiceURL: strings.TrimRight(env("MODEL_SERVICE_URL", "http://localhost:8001"), "/"),
		ModelTimeout:    modelTimeout,
		MaxTextLength:   maxTextLength,
		RedisURL:        env("REDIS_URL", "redis://localhost:6379/0"),
		SessionTTL:      time.Duration(sessionTTL) * time.Second,
		CORSOrigins:     origins,
	}, nil
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
