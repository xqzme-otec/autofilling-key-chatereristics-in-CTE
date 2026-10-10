package config

import (
	"reflect"
	"testing"
	"time"
)

var keys = []string{"MODEL_SERVICE_URL", "MODEL_TIMEOUT_S", "MAX_TEXT_LENGTH", "REDIS_URL", "SESSION_TTL_S", "CORS_ORIGINS"}

func TestLoad(t *testing.T) {
	for _, k := range keys {
		t.Setenv(k, "")
	}
	cfg, err := Load()
	if err != nil {
		t.Fatal(err)
	}
	want := Config{ModelServiceURL: "http://localhost:8001", ModelTimeout: 30 * time.Second, MaxTextLength: 2000,
		RedisURL: "redis://localhost:6379/0", SessionTTL: time.Hour, CORSOrigins: []string{"*"}}
	if !reflect.DeepEqual(cfg, want) {
		t.Errorf("defaults: got %+v, want %+v", cfg, want)
	}

	t.Setenv("MODEL_SERVICE_URL", "http://model:8001/")
	t.Setenv("MODEL_TIMEOUT_S", "2.5")
	t.Setenv("CORS_ORIGINS", " http://a , ,http://b")
	cfg, err = Load()
	if err != nil {
		t.Fatal(err)
	}
	if cfg.ModelServiceURL != "http://model:8001" || cfg.ModelTimeout != 2500*time.Millisecond ||
		!reflect.DeepEqual(cfg.CORSOrigins, []string{"http://a", "http://b"}) {
		t.Errorf("overrides: got %+v", cfg)
	}

	for key, bad := range map[string]string{"MODEL_TIMEOUT_S": "NaN", "MAX_TEXT_LENGTH": "2.5", "SESSION_TTL_S": "0"} {
		t.Run(key, func(t *testing.T) {
			t.Setenv(key, bad)
			if _, err := Load(); err == nil {
				t.Errorf("%s=%q: want error", key, bad)
			}
		})
	}
}
