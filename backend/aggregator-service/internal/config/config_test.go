package config

import (
	"reflect"
	"testing"
	"time"

	"go.uber.org/zap/zapcore"
)

var keys = []string{"LOG_LEVEL", "LISTEN_ADDR", "SHUTDOWN_TIMEOUT_S", "READ_HEADER_TIMEOUT_S", "MAX_BODY_BYTES",
	"REDIS_PING_TIMEOUT_S", "REDIS_OP_TIMEOUT_S", "MODEL_SERVICE_URL", "MODEL_TIMEOUT_S", "MAX_TEXT_LENGTH", "MAX_BATCH_ITEMS", "BATCH_CONCURRENCY", "REDIS_URL", "SESSION_TTL_S", "CORS_ORIGINS"}

func TestLoad(t *testing.T) {
	for _, k := range keys {
		t.Setenv(k, "")
	}
	cfg, err := Load()
	if err != nil {
		t.Fatal(err)
	}
	want := Config{LogLevel: zapcore.InfoLevel, ListenAddr: ":8000", ShutdownTimeout: 8 * time.Second, ReadHeaderTimeout: 10 * time.Second,
		MaxBodyBytes: 1 << 20, ModelServiceURL: "http://localhost:8001", ModelTimeout: 30 * time.Second,
		MaxTextLength: 2000, MaxBatchItems: 50, BatchConcurrency: 4, RedisURL: "redis://localhost:6379/0", RedisPingTimeout: 5 * time.Second,
		RedisOpTimeout: 3 * time.Second, SessionTTL: time.Hour, CORSOrigins: []string{"*"}}
	if !reflect.DeepEqual(cfg, want) {
		t.Errorf("defaults: got %+v, want %+v", cfg, want)
	}

	t.Setenv("MODEL_SERVICE_URL", "http://model:8001/")
	t.Setenv("MODEL_TIMEOUT_S", "2.5")
	t.Setenv("CORS_ORIGINS", " http://a , ,http://b")
	t.Setenv("LOG_LEVEL", "DEBUG") // the Python services spell levels in upper case
	cfg, err = Load()
	if err != nil {
		t.Fatal(err)
	}
	if cfg.ModelServiceURL != "http://model:8001" || cfg.ModelTimeout != 2500*time.Millisecond ||
		cfg.LogLevel != zapcore.DebugLevel ||
		!reflect.DeepEqual(cfg.CORSOrigins, []string{"http://a", "http://b"}) {
		t.Errorf("overrides: got %+v", cfg)
	}

	for key, bad := range map[string]string{"MODEL_TIMEOUT_S": "NaN", "MAX_TEXT_LENGTH": "2.5", "SESSION_TTL_S": "0",
		"SHUTDOWN_TIMEOUT_S": "-1", "MAX_BODY_BYTES": "x", "REDIS_OP_TIMEOUT_S": "0", "LOG_LEVEL": "verbose", "MAX_BATCH_ITEMS": "0", "BATCH_CONCURRENCY": "-2"} {
		t.Run(key, func(t *testing.T) {
			t.Setenv(key, bad)
			if _, err := Load(); err == nil {
				t.Errorf("%s=%q: want error", key, bad)
			}
		})
	}
}
