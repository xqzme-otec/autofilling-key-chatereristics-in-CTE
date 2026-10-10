// Package app wires the layers together and runs the aggregator.
package app

import (
	"context"
	"log/slog"
	"net/http"
	"time"

	"github.com/xqzme-otec/autofilling-key-chatereristics-in-CTE/backend/internal/client/model"
	"github.com/xqzme-otec/autofilling-key-chatereristics-in-CTE/backend/internal/config"
	"github.com/xqzme-otec/autofilling-key-chatereristics-in-CTE/backend/internal/database"
	"github.com/xqzme-otec/autofilling-key-chatereristics-in-CTE/backend/internal/repository"
	"github.com/xqzme-otec/autofilling-key-chatereristics-in-CTE/backend/internal/service"
	httptransport "github.com/xqzme-otec/autofilling-key-chatereristics-in-CTE/backend/internal/transport/http"
)

const (
	listenAddr = ":8000"
	// shutdownTimeout lets in-flight requests finish; it stays under Docker's 10 s stop timeout.
	shutdownTimeout = 8 * time.Second
)

// Run serves the API until ctx is cancelled (then shuts down gracefully) or the server fails.
func Run(ctx context.Context) error {
	cfg, err := config.Load()
	if err != nil {
		return err
	}
	rdb, err := database.NewRedis(ctx, cfg.RedisURL)
	if err != nil {
		return err
	}
	defer rdb.Close()

	analysis := service.NewAnalysis(
		repository.NewRedisSessionRepository(rdb, cfg.SessionTTL),
		model.NewClient(cfg.ModelServiceURL, cfg.ModelTimeout),
		cfg.MaxTextLength,
	)
	srv := &http.Server{
		Addr:              listenAddr,
		Handler:           httptransport.NewRouter(analysis, cfg.CORSOrigins),
		ReadHeaderTimeout: 10 * time.Second,
	}

	errc := make(chan error, 1)
	go func() { errc <- srv.ListenAndServe() }()
	slog.Info("aggregator listening", "addr", listenAddr, "model", cfg.ModelServiceURL)

	select {
	case err := <-errc:
		return err
	case <-ctx.Done():
	}
	slog.Info("shutting down")
	shutdownCtx, cancel := context.WithTimeout(context.Background(), shutdownTimeout)
	defer cancel()
	return srv.Shutdown(shutdownCtx)
}
