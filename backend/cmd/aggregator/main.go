// Command aggregator is the single entry point for the frontend: it validates and preprocesses
// the text, calls the stateless model service and keeps the clarifying dialog session in Redis.
package main

import (
	"context"
	"log/slog"
	"os"
	"os/signal"
	"syscall"

	"github.com/xqzme-otec/autofilling-key-chatereristics-in-CTE/backend/internal/app"
)

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	err := app.Run(ctx)
	stop()
	if err != nil {
		slog.Error("aggregator stopped", "err", err)
		os.Exit(1)
	}
}
