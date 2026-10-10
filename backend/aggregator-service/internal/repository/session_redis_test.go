package repository_test

import (
	"context"
	"encoding/json"
	"errors"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/redis/go-redis/v9"

	"github.com/xqzme-otec/autofilling-key-chatereristics-in-CTE/backend/aggregator-service/internal/domain"
	"github.com/xqzme-otec/autofilling-key-chatereristics-in-CTE/backend/aggregator-service/internal/repository"
)

func TestRedisSessionRepository(t *testing.T) {
	mr := miniredis.RunT(t)
	repo := repository.NewRedisSessionRepository(
		redis.NewClient(&redis.Options{Addr: mr.Addr(), MaxRetries: -1}), time.Hour, time.Second)
	ctx := context.Background()

	if _, err := repo.Get(ctx, "missing"); !errors.Is(err, repository.ErrSessionNotFound) {
		t.Fatalf("missing session: got %v, want ErrSessionNotFound", err)
	}

	text := "стул"
	want := &domain.Session{
		ID: "s1", Text: text,
		Result:    domain.AnalysisResult{Input: &text, Text: &text},
		Answers:   []domain.Answer{{Type: domain.AnswerProduct, Value: "Стул"}},
		CreatedAt: time.Date(2026, 10, 10, 0, 0, 0, 0, time.UTC),
		UpdatedAt: time.Date(2026, 10, 10, 0, 1, 0, 0, time.UTC),
	}
	if err := repo.Save(ctx, want); err != nil {
		t.Fatal(err)
	}
	if ttl := mr.TTL("session:s1"); ttl != time.Hour {
		t.Errorf("TTL after save = %v, want 1h", ttl)
	}

	mr.FastForward(30 * time.Minute)
	got, err := repo.Get(ctx, "s1")
	if err != nil {
		t.Fatal(err)
	}
	if g, w := mustJSON(t, got), mustJSON(t, want); g != w {
		t.Errorf("got  %s\nwant %s", g, w)
	}
	if ttl := mr.TTL("session:s1"); ttl != time.Hour {
		t.Errorf("TTL after read = %v, want renewed to 1h", ttl)
	}

	mr.FastForward(2 * time.Hour)
	if _, err := repo.Get(ctx, "s1"); !errors.Is(err, repository.ErrSessionNotFound) {
		t.Errorf("expired session: got %v, want ErrSessionNotFound", err)
	}

	mr.Close()
	var de *domain.Error
	if err := repo.Save(ctx, want); !errors.As(err, &de) || de.Code != domain.CodeStorageUnavailable {
		t.Errorf("redis down: got %v, want storage_unavailable", err)
	}
}

func mustJSON(t *testing.T, v any) string {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}
