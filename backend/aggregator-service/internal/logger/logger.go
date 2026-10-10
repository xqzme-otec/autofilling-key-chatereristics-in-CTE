// Package logger builds the zap logger of the aggregator and carries the per-request logger,
// annotated with the request ID, through a context.
package logger

import (
	"context"

	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
)

// RequestIDHeader carries the request ID from the client to the aggregator and on to the model
// service, so the log lines of one request can be found in both services.
const RequestIDHeader = "X-Request-ID"

// New builds the service logger: plain-text lines on stderr.
func New(level zapcore.Level) (*zap.Logger, error) {
	cfg := zap.NewProductionConfig()
	cfg.Level = zap.NewAtomicLevelAt(level)
	cfg.Encoding = "console"
	cfg.EncoderConfig.EncodeTime = zapcore.ISO8601TimeEncoder
	cfg.EncoderConfig.EncodeLevel = zapcore.CapitalLevelEncoder
	cfg.Sampling = nil           // never drop request lines
	cfg.DisableStacktrace = true // errors are values: the stack of the log call adds nothing
	return cfg.Build()
}

type ctxKey struct{}

type request struct {
	id  string
	log *zap.Logger
}

// WithRequestID returns a copy of ctx that carries the request ID and the global logger
// annotated with it.
func WithRequestID(ctx context.Context, id string) context.Context {
	return context.WithValue(ctx, ctxKey{}, request{id: id, log: zap.L().With(zap.String("request_id", id))})
}

// RequestID returns the ID of the request ctx belongs to, or "" outside a request.
func RequestID(ctx context.Context) string {
	r, _ := ctx.Value(ctxKey{}).(request)
	return r.id
}

// FromContext returns the logger of the request ctx belongs to, or the global logger outside a request.
func FromContext(ctx context.Context) *zap.Logger {
	if r, ok := ctx.Value(ctxKey{}).(request); ok {
		return r.log
	}
	return zap.L()
}
