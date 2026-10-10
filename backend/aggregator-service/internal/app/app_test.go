package app_test

import (
	"context"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"

	"github.com/xqzme-otec/autofilling-key-chatereristics-in-CTE/backend/aggregator-service/internal/app"
	"github.com/xqzme-otec/autofilling-key-chatereristics-in-CTE/backend/aggregator-service/internal/config"
)

// After ctx is cancelled the server stops accepting connections, but a request that is already
// in flight still gets its response, and Run returns nil.
func TestRunDrainsInFlightRequests(t *testing.T) {
	inModel, release := make(chan struct{}), make(chan struct{})
	modelSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		close(inModel)
		select {
		case <-release:
		case <-r.Context().Done(): // the test failed and the aggregator gave up
		}
		io.WriteString(w, `{"input": "стул", "text": "стул"}`)
	}))
	defer modelSrv.Close()

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	addr := ln.Addr().String()
	ln.Close()
	cfg := config.Config{ListenAddr: addr, ShutdownTimeout: 5 * time.Second, ReadHeaderTimeout: time.Second,
		MaxBodyBytes: 1 << 20, ModelServiceURL: modelSrv.URL, ModelTimeout: 10 * time.Second, MaxTextLength: 100,
		RedisURL: "redis://" + miniredis.RunT(t).Addr(), RedisPingTimeout: time.Second, RedisOpTimeout: time.Second,
		SessionTTL: time.Hour}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	stopped := make(chan error, 1)
	go func() { stopped <- app.Run(ctx, cfg) }()
	listening := func() bool {
		conn, err := net.Dial("tcp", addr)
		if err == nil {
			conn.Close()
		}
		return err == nil
	}
	waitFor(t, "the server to start", listening)

	status := make(chan int, 1)
	go func() {
		resp, err := http.Post("http://"+addr+"/api/v1/analyze", "application/json", strings.NewReader(`{"text": "стул"}`))
		if err != nil {
			status <- 0
			return
		}
		resp.Body.Close()
		status <- resp.StatusCode
	}()
	select {
	case <-inModel:
	case got := <-status:
		t.Fatalf("request finished before the shutdown: %d", got)
	case <-time.After(5 * time.Second):
		t.Fatal("request did not reach the model")
	}

	cancel()
	waitFor(t, "the listener to close", func() bool { return !listening() })
	close(release)

	if got := <-status; got != http.StatusOK {
		t.Errorf("in-flight request: got %d, want 200", got)
	}
	if err := <-stopped; err != nil {
		t.Errorf("Run: %v", err)
	}
}

func waitFor(t *testing.T, what string, cond func() bool) {
	t.Helper()
	for deadline := time.Now().Add(5 * time.Second); !cond(); time.Sleep(10 * time.Millisecond) {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s", what)
		}
	}
}
