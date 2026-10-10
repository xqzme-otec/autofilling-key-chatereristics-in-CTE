// Package app wires the layers together and runs the aggregator.
package app

import (
	"context"
	"fmt"
	"net/http"

	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"

	"github.com/xqzme-otec/autofilling-key-chatereristics-in-CTE/backend/aggregator-service/internal/client/model"
	"github.com/xqzme-otec/autofilling-key-chatereristics-in-CTE/backend/aggregator-service/internal/config"
	"github.com/xqzme-otec/autofilling-key-chatereristics-in-CTE/backend/aggregator-service/internal/database"
	"github.com/xqzme-otec/autofilling-key-chatereristics-in-CTE/backend/aggregator-service/internal/repository"
	"github.com/xqzme-otec/autofilling-key-chatereristics-in-CTE/backend/aggregator-service/internal/service"
	httptransport "github.com/xqzme-otec/autofilling-key-chatereristics-in-CTE/backend/aggregator-service/internal/transport/http"
)

// Run serves the API until ctx is cancelled, then shuts down gracefully: the server stops
// accepting connections, in-flight requests get cfg.ShutdownTimeout to finish, and only then
// Redis is closed. Logs go to the global zap logger.
func Run(ctx context.Context, cfg config.Config) error {
	log := zap.L()
	rdb, err := database.NewRedis(ctx, cfg.RedisURL, cfg.RedisPingTimeout)
	if err != nil {
		return err
	}
	defer rdb.Close()

	analysis := service.NewAnalysis(
		repository.NewRedisSessionRepository(rdb, cfg.SessionTTL, cfg.RedisOpTimeout),
		model.NewClient(cfg.ModelServiceURL, cfg.ModelTimeout),
		service.Limits{MaxTextLength: cfg.MaxTextLength, MaxBatchItems: cfg.MaxBatchItems,
			BatchConcurrency: cfg.BatchConcurrency},
	)
	serverLog, _ := zap.NewStdLogAt(log, zapcore.ErrorLevel) // fails only for an invalid level
	srv := &http.Server{
		Addr:              cfg.ListenAddr,
		Handler:           httptransport.NewRouter(analysis, cfg.CORSOrigins, cfg.MaxBodyBytes),
		ReadHeaderTimeout: cfg.ReadHeaderTimeout,
		ErrorLog:          serverLog, // e.g. recovered handler panics
	}

	errc := make(chan error, 1)
	go func() { errc <- srv.ListenAndServe() }()
	log.Info("aggregator listening", zap.String("addr", cfg.ListenAddr), zap.String("model", cfg.ModelServiceURL))

	select {
	case err := <-errc:
		return err
	case <-ctx.Done():
	}
	log.Info("shutting down", zap.Stringer("timeout", cfg.ShutdownTimeout))
	shutdownCtx, cancel := context.WithTimeout(context.Background(), cfg.ShutdownTimeout)
	defer cancel()
	if err := srv.Shutdown(shutdownCtx); err != nil {
		srv.Close() // drop the requests that did not finish in time
		return fmt.Errorf("graceful shutdown: %w", err)
	}
	log.Info("aggregator stopped")
	return nil
}
