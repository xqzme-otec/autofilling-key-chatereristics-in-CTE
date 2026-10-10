// Command aggregator is the single entry point for the frontend: it validates the text, calls the
// stateless model service and keeps the clarifying dialog session in Redis.
package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"go.uber.org/zap"

	"github.com/xqzme-otec/autofilling-key-chatereristics-in-CTE/backend/aggregator-service/internal/app"
	"github.com/xqzme-otec/autofilling-key-chatereristics-in-CTE/backend/aggregator-service/internal/config"
	"github.com/xqzme-otec/autofilling-key-chatereristics-in-CTE/backend/aggregator-service/internal/logger"
)

func main() {
	cfg, err := config.Load()
	if err != nil {
		// The logger is built from cfg, so a config error can only be printed as is.
		fmt.Fprintln(os.Stderr, "aggregator: invalid config:", err)
		os.Exit(1)
	}
	log, err := logger.New(cfg.LogLevel)
	if err != nil {
		fmt.Fprintln(os.Stderr, "aggregator: create logger:", err)
		os.Exit(1)
	}
	zap.ReplaceGlobals(log)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	// Once the first signal starts the graceful shutdown, the default handling is back:
	// a second Ctrl+C kills the process at once.
	context.AfterFunc(ctx, stop)
	err = app.Run(ctx, cfg)
	stop()
	if err != nil {
		log.Error("aggregator failed", zap.Error(err))
	}
	_ = log.Sync() // the error is ignored: syncing a terminal or pipe fails harmlessly
	if err != nil {
		os.Exit(1)
	}
}
